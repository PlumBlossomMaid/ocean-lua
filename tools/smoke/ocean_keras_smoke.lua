-- tools/smoke/ocean_keras_smoke.lua -- ocean-lua Keras 模式端到端补验
--
-- 判据:ocean.Model{net=...}:prepare{optimizer, loss, metrics} +
-- :fit{loader, epochs, val_loader} 全链路 —— Keras 风格 quick-prototype
-- 路径(overview.md §6.3 模式②)。用 MNIST 子集(2000 训练样本,提速)
-- 验证:
--   1. prepare 装配 optimizer/loss/metrics
--   2. fit 驱动训练循环(loss 下降)
--   3. val_loader 传入后每 epoch 跑验证,MetricCollection 结果
--      (val_acc / val_loss)写入日志
--   4. trainer 返回给调用方,logger 可查
--
-- 数据管线同 ocean_mnist_smoke(paddle.io.TensorDataset/DataLoader
-- 公开 API);唯一 paddle_core 使用 = 底层 IDX 读取。
--
-- Run (from the paddle-lua repo root):
--   PATH=<paddle runtime dll dirs>;$PATH luajit E:\code\ocean-lua\tools\smoke\ocean_keras_smoke.lua

package.cpath = "./build-test/?.dll;" .. package.cpath
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.path = "E:\\code\\ocean-lua/lua/?.lua;E:\\code\\ocean-lua/lua/?/init.lua;"
               .. package.path
package.path = "E:\\code\\metrics-lua/lua/?.lua;E:\\code\\metrics-lua/lua/?/init.lua;"
               .. package.path

local paddle_core = require "paddle_core"
paddle_core.init()
require "paddle.tensor"
local paddle = require "paddle"
local ocean = require "ocean"
local metrics = require "metrics"
local class = require "pl.class"

local DATA_DIR = os.getenv("MNIST_DATA_DIR") or "build-test/mnist-data"

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

-- 读 IDX + 造张量(同 ocean_mnist_smoke)
local function be_u32(s, off)
  return s:byte(off) * 16777216 + s:byte(off + 1) * 65536
       + s:byte(off + 2) * 256 + s:byte(off + 3)
end
local function read_idx(path, want_magic)
  local f = assert(io.open(path, "rb"), "cannot open " .. path)
  local all = f:read("*a")
  f:close()
  local magic = be_u32(all, 1)
  assert(magic == want_magic, ("%s: bad magic 0x%08x"):format(path, magic))
  local num = be_u32(all, 5)
  if want_magic == 2051 then
    local rows = be_u32(all, 9)
    local cols = be_u32(all, 13)
    return num, rows, cols, string.sub(all, 17)
  end
  return num, nil, nil, string.sub(all, 9)
end

local CPU_PLACE = paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)

local function make_images(rows, cols, payload)
  local u8 = paddle_core.from_host_bytes(
    payload, { math.floor(#payload / (rows * cols)), 1, rows, cols },
    paddle_core.PD_DTYPE_UINT8, CPU_PLACE)
  local f32 = paddle_core.pd_op_cast(u8, paddle_core.PD_DTYPE_FLOAT32)
  return paddle_core.pd_op_scale(f32, 2 / 255, -1, 1)
end
local function make_labels(payload)
  local u8 = paddle_core.from_host_bytes(
    payload, { #payload }, paddle_core.PD_DTYPE_UINT8, CPU_PLACE)
  local i64 = paddle_core.pd_op_cast(u8, paddle_core.PD_DTYPE_INT64)
  local ones = paddle_core.pd_op_full(
    { #payload }, 1, paddle_core.PD_DTYPE_INT64, CPU_PLACE)
  return paddle_core.pd_op_add(i64, ones)
end

-- 取 tensor 前 n 行(公开 API:paddle.slice 或底层;用 pd_op_slice)
local function head(t, n)
  return paddle_core.pd_op_slice(t, { 0 }, { 0 }, { n }, { 1 }, {})
end

local tr_n, tr_rows, tr_cols, tr_img =
  read_idx(DATA_DIR .. "/train-images-idx3-ubyte", 2051)
local _, _, _, tr_label =
  read_idx(DATA_DIR .. "/train-labels-idx1-ubyte", 2049)
local te_n, te_rows, te_cols, te_img =
  read_idx(DATA_DIR .. "/t10k-images-idx3-ubyte", 2051)
local _, _, _, te_label =
  read_idx(DATA_DIR .. "/t10k-labels-idx1-ubyte", 2049)

-- 子集:训练 2000 / 测试 500
local SUB = 2000
local tr_x = head(make_images(tr_rows, tr_cols, tr_img), SUB)
local tr_y = head(make_labels(tr_label), SUB)
local te_x = head(make_images(te_rows, te_cols, te_img), 500)
local te_y = head(make_labels(te_label), 500)

local tr_ds = paddle.io.TensorDataset{ tensors = { tr_x, tr_y } }
local te_ds = paddle.io.TensorDataset{ tensors = { te_x, te_y } }
local BATCH = 64
local train_loader = paddle.io.DataLoader{ dataset = tr_ds, batch_size = BATCH }
local val_loader = paddle.io.DataLoader{ dataset = te_ds, batch_size = BATCH }

-- 网络(小 LeNet:Conv2D 8 通道降计算)
local nn = paddle.nn
local net = nn.Sequential{
  nn.Conv2D{ in_channels = 1, out_channels = 8, kernel_size = 3, padding = 1 },
  nn.ReLU(),
  nn.MaxPool2D{ kernel_size = 2 },
  nn.Conv2D{ in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1 },
  nn.ReLU(),
  nn.MaxPool2D{ kernel_size = 2 },
  nn.Flatten(),
  nn.Linear{ in_features = 16 * 7 * 7, out_features = 10 },
}

-- Linear 显式 gaussian(同 mnist.lua)
local old_g = paddle_core.has_grad()
paddle_core.set_has_grad(false)
for _, p in ipairs(net:parameters()) do
  local s = p:shape()
  if #s == 2 then
    local fan_in = s[1]
    paddle_core.pd_op_gaussian_inplace(p, 0.0, (2 / fan_in) ^ 0.5, 0)
  end
end
paddle_core.set_has_grad(old_g)

-- ---------------------------------------------------------------------
-- Keras Model:prepare + fit
-- ---------------------------------------------------------------------

local km = ocean.Model{ net = net }
local opt = paddle.optimizer.Adam{ parameters = net:parameters(),
                                   learning_rate = 0.001 }
local loss_fn = function(pred, target)
  return paddle.nn.functional.cross_entropy{ input = pred, label = target,
                                             reduction = "mean" }
end
km:prepare{
  optimizer = opt,
  loss = loss_fn,
  metrics = { acc = metrics.Accuracy{ multiclass = true, num_classes = 10 } },
}

-- 装配断言
check(km._optimizer == opt, "Keras prepare binds optimizer")
check(km._metrics ~= nil and km._metrics.log_dict ~= nil,
      "Keras prepare wraps metrics into MetricCollection")

-- fit 全链路(1 epoch,val 每 epoch 上报)
local trainer = km:fit{ loader = train_loader, epochs = 1,
                        val_loader = val_loader }

-- 训练 loss 已记录
local tl = trainer._logger:epoch_metric("train_loss", 1)
check(tl ~= nil and tl > 0, "Keras fit recorded train_loss")

-- 验证指标上报(val_acc / val_loss)。0.9 是收敛后预期;1 epoch 子集
-- (2000 样本)训练只要求显著高于随机 0.1 —— 验证的是"上报机制",
-- 收敛判据已在 ocean_mnist_smoke(2 epoch 全量 acc>0.95)覆盖。
local va = trainer:monitor_value("val_acc")
local vl = trainer:monitor_value("val_loss")
check(va ~= nil and va > 0.5, ("Keras val_acc reported (%s > 0.5)"):format(
  tostring(va)))
check(vl ~= nil and vl > 0, "Keras val_loss reported")

-- 2 epoch:train_loss 下降
local km2 = ocean.Model{ net = nn.Linear{ in_features = 4, out_features = 1 } }
local opt2 = paddle.optimizer.SGD{ parameters = km2.net:parameters(),
                                   learning_rate = 0.05 }
-- 简单的回归:Keras 模式 loss 下降
local loss2 = function(pred, target)
  return paddle.nn.functional.mse_loss{ input = pred, label = target,
                                        reduction = "mean" }
end
km2:prepare{ optimizer = opt2, loss = loss2 }
-- 造小回归数据:固定 x=[1,1,1,1] y=[1.0],验证 loss 确实下降(梯度非零)
local F32 = paddle_core.PD_DTYPE_FLOAT32
local reg_rows = {}
for i = 1, 32 do
  local x = paddle_core.pd_op_full({ 4 }, 1.0, F32, CPU_PLACE)
  local y = paddle_core.pd_op_full({ 1 }, 1.0, F32, CPU_PLACE)
  reg_rows[i] = { x, y }
end
local RegDS = class(paddle.io.Dataset)
function RegDS:_init(rows) self.rows = rows end
function RegDS:len() return #self.rows end
function RegDS:get(i) return self.rows[i] end
local reg_loader = paddle.io.DataLoader{ dataset = RegDS(reg_rows),
                                         batch_size = 16 }
local t2 = km2:fit{ loader = reg_loader, epochs = 2 }
local rl1 = t2._logger:epoch_metric("train_loss", 1)
local rl2 = t2._logger:epoch_metric("train_loss", 2)
check(rl1 ~= nil and rl2 ~= nil and rl2 < rl1,
      ("Keras regression loss decreased 1->2 (%.4g -> %.4g)"):format(
        rl1 or -1, rl2 or -1))

print(("SMOKE PASS: ocean-lua Keras mode OK (%d assertions)"):format(ntests))
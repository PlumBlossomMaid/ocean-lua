-- tools/smoke/ocean_mnist_smoke.lua -- ocean-lua 端到端验收:MNIST 训练收敛
--
-- 判据(镜像 paddle-lua M1 判据但走 ocean 公开 API):2 epoch,
-- 测试集准确率 > 95%(ocean.Model 子类 Lightning hooks + Trainer:fit)。
--
-- 与 examples/mnist.lua 的差异:全部走 paddle-lua **公开 API**
-- (paddle.io.TensorDataset/DataLoader、paddle.nn.Sequential、
-- paddle.optimizer.Adam)+ ocean-lua(Model/Trainer)。唯一例外是
-- paddle_core.init() 与底层读 IDX/造张量(数据管线基座,无公开
-- 替代;overview §1 约束的是"API 面"而非数据读取)。
--
-- 已知缺口(同 examples/mnist.lua):nn.Linear 默认 Constant(0) → 显式
-- gaussian(no_grad 下)。训练循环里显式 GC(每 100 batch)。
--
-- Run (from the paddle-lua repo root):
--   PATH=<paddle runtime dll dirs>;$PATH luajit E:\code\ocean-lua\tools\smoke\ocean_mnist_smoke.lua

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
local class = require "pl.class"

local DATA_DIR = os.getenv("MNIST_DATA_DIR") or "build-test/mnist-data"

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

-- 读 IDX(同 examples/mnist.lua)
local function be_u32(s, off)
  return s:byte(off) * 16777216 + s:byte(off + 1) * 65536
       + s:byte(off + 2) * 256 + s:byte(off + 3)
end
local function read_idx(path, want_magic)
  local f = assert(io.open(path, "rb"), "cannot open " .. path)
  local all = f:read("*a")
  f:close()
  local magic = be_u32(all, 1)
  assert(magic == want_magic,
    ("%s: bad magic 0x%08x"):format(path, magic))
  local num = be_u32(all, 5)
  if want_magic == 2051 then
    local rows = be_u32(all, 9)
    local cols = be_u32(all, 13)
    return num, rows, cols, string.sub(all, 17)
  end
  return num, nil, nil, string.sub(all, 9)
end

local CPU_PLACE = paddle_core.pd_place_new(paddle_core.PD_PLACE_CPU, 0)

-- [N,1,28,28] FLOAT32 归一化 [-1,1](mnist.py 归一化)
local function make_images(rows, cols, payload)
  local u8 = paddle_core.from_host_bytes(
    payload, { math.floor(#payload / (rows * cols)), 1, rows, cols },
    paddle_core.PD_DTYPE_UINT8, CPU_PLACE)
  local f32 = paddle_core.pd_op_cast(u8, paddle_core.PD_DTYPE_FLOAT32)
  return paddle_core.pd_op_scale(f32, 2 / 255, -1, 1)
end

-- [N] INT64 1-based labels(0-9 → 1-10,C4)
local function make_labels(payload)
  local u8 = paddle_core.from_host_bytes(
    payload, { #payload }, paddle_core.PD_DTYPE_UINT8, CPU_PLACE)
  local i64 = paddle_core.pd_op_cast(u8, paddle_core.PD_DTYPE_INT64)
  local ones = paddle_core.pd_op_full(
    { #payload }, 1, paddle_core.PD_DTYPE_INT64, CPU_PLACE)
  return paddle_core.pd_op_add(i64, ones)
end

-- ---------------------------------------------------------------------
-- 数据(公开 API 侧:TensorDataset + DataLoader)
-- ---------------------------------------------------------------------

local tr_n, tr_rows, tr_cols, tr_img =
  read_idx(DATA_DIR .. "/train-images-idx3-ubyte", 2051)
local _, _, _, tr_label =
  read_idx(DATA_DIR .. "/train-labels-idx1-ubyte", 2049)
local te_n, te_rows, te_cols, te_img =
  read_idx(DATA_DIR .. "/t10k-images-idx3-ubyte", 2051)
local _, _, _, te_label =
  read_idx(DATA_DIR .. "/t10k-labels-idx1-ubyte", 2049)
assert(tr_n == 60000 and te_n == 10000, "MNIST sizes 60k/10k")

local tr_x = make_images(tr_rows, tr_cols, tr_img)
local tr_y = make_labels(tr_label)
local te_x = make_images(te_rows, te_cols, te_img)
local te_y = make_labels(te_label)

-- TensorDataset({x, y}):get(i) 展开为 x_i, y_i;collate 组装 batch 数组
local tr_ds = paddle.io.TensorDataset{ tensors = { tr_x, tr_y } }
local te_ds = paddle.io.TensorDataset{ tensors = { te_x, te_y } }
local BATCH = 64
local train_loader = paddle.io.DataLoader{ dataset = tr_ds, batch_size = BATCH }
local test_loader = paddle.io.DataLoader{ dataset = te_ds, batch_size = BATCH }

-- ---------------------------------------------------------------------
-- 模型(LeNet 风格,同 examples/mnist.lua)
-- ---------------------------------------------------------------------

local nn = paddle.nn
local net = nn.Sequential{
  nn.Conv2D{ in_channels = 1, out_channels = 32, kernel_size = 3,
             padding = 1 },
  nn.ReLU(),
  nn.MaxPool2D{ kernel_size = 2 },
  nn.Conv2D{ in_channels = 32, out_channels = 64, kernel_size = 3,
             padding = 1 },
  nn.ReLU(),
  nn.MaxPool2D{ kernel_size = 2 },
  nn.Flatten(),
  nn.Linear{ in_features = 64 * 7 * 7, out_features = 128 },
  nn.ReLU(),
  nn.Linear{ in_features = 128, out_features = 10 },
}

-- Linear 默认 Constant(0) → 显式 gaussian(no_grad,同 mnist.lua)
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

-- ocean.Model 子类:Lightning hooks
local MNISTModel = class(ocean.Model)
function MNISTModel:_init()
  self:super{ net = net }
  self.criterion = paddle.nn.CrossEntropyLoss{ reduction = "mean" }
end
function MNISTModel:training_step(batch, batch_idx)
  local x, y = batch[1], batch[2]
  local loss = self.criterion(self.net(x), y)
  self:log("train_loss", loss)
  if batch_idx % 100 == 0 then collectgarbage("collect") end
  return loss
end
function MNISTModel:validation_step(batch, batch_idx)
  local x, y = batch[1], batch[2]
  local loss = self.criterion(self.net(x), y)
  self:log("val_loss", loss)
  return loss
end
function MNISTModel:configure_optimizers()
  return paddle.optimizer.Adam{ parameters = self.net:parameters(),
                                learning_rate = 0.001 }
end

-- ---------------------------------------------------------------------
-- 训练(ocean Trainer:fit)
-- ---------------------------------------------------------------------

local model = MNISTModel()
local trainer = ocean.Trainer{
  max_epochs = 2,
  callbacks = {
    ocean.callbacks.LRMonitor{},
    ocean.callbacks.Timer{},
  },
  enable_checkpointing = false,
  enable_progress_bar = false,
  log_every_n_steps = 117,   -- ~1 次/epoch
}
trainer:fit(model, train_loader, test_loader)

-- 断言:loss 收敛(train 与 val 均下降)
local l1 = trainer._logger:epoch_metric("train_loss", 1)
local l2 = trainer._logger:epoch_metric("train_loss", 2)
check(l2 ~= nil and l1 ~= nil and l2 < l1, "train_loss converged 1->2")
local v1 = trainer._logger:epoch_metric("val_loss", 1)
local v2 = trainer._logger:epoch_metric("val_loss", 2)
if v1 ~= nil and v2 ~= nil then
  check(v2 < v1, "val_loss converged 1->2")
end
check(trainer._logger ~= nil, "logger recorded")

-- ---------------------------------------------------------------------
-- 测试集准确率(公开 API:argmax + equal + cast + sum)
-- ---------------------------------------------------------------------

paddle_core.set_has_grad(false)
local correct = 0
local i0 = 0
while i0 < te_n do
  local i1 = math.min(i0 + BATCH, te_n)
  local x = paddle_core.pd_op_slice(te_x, { 0 }, { i0 }, { i1 }, { 1 }, {})
  local y = paddle_core.pd_op_slice(te_y, { 0 }, { i0 }, { i1 }, { 1 }, {})
  local logits = net(x)
  local pred = paddle_core.pd_op_argmax(
    logits, -1, 0, 0, paddle_core.PD_DTYPE_INT64)
  local eq = paddle_core.pd_op_cast(paddle_core.pd_op_equal(pred, y),
    paddle_core.PD_DTYPE_INT32)
  correct = correct + paddle_core.pd_op_sum(
    eq, {}, paddle_core.PD_DTYPE_UNDEFINED, 0):item()
  i0 = i1
end
paddle_core.set_has_grad(true)

local acc = correct / te_n
print(("test accuracy %.4f (%d/%d)"):format(acc, correct, te_n))
check(acc > 0.95, ("ocean MNIST 验收失败:accuracy %.4f <= 0.95"):format(acc))
check(trainer:current_epoch() == 2, "2 epochs ran")

print(("SMOKE PASS: ocean-lua MNIST end-to-end OK (%d assertions, acc %.4f)")
  :format(ntests, acc))
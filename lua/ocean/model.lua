-- lua/ocean/model.lua -- 双模式 Model(Lightning hooks + Keras prepare/fit)
--
-- ① Lightning 风格:用户子类覆写 training_step/validation_step/
--    configure_optimizers/configure_loss 等 hooks,Trainer 驱动循环。
-- ② Keras 风格:Model{net=...}:prepare{optimizer, loss, metrics} +
--    :fit{loader, epochs},面向快速原型。
--
-- Model 包装 paddle.nn.Layer(或裸 Sequential),负责:
--   * 前向 forward(batch)(Keras 的 net 调用)
--   * 优化器/损失/metrics 的持有与 step 调度
--   * log/current_lr/parameters/state_dict 透传
-- 纯 Lua 5.1 + pl.class(与 metrics-lua 同基座)。

local class = require "pl.class"
local paddle = require "paddle"
local argrule = require "argrule"
local rule = require "argrule.rule"
local accel = require "ocean.accelerators"
local metrics_lib = require "metrics"

local Model = class()

-- 判 pl.class 实例(Layer 或 Model)
local function is_pl_class(v)
  return type(v) == "table" and type(getmetatable(v)) == "table"
         and type(getmetatable(v).__index) == "table"
end

-- Keras 构造:Model{net=..., accelerator=...}
Model._init = rule {
  { name = "self", type = argrule.any },
  { name = "net", type = argrule.any, opt = true, default = nil,
    doc = "paddle.nn.Layer or Sequential (Keras style)" },
  { name = "accelerator", type = argrule.any, opt = true, default = "cpu",
    doc = "accelerator spec: 'cpu'|'cuda'|Accelerator instance" },
  { name = "strategy", type = argrule.any, opt = true, default = "single_device",
    doc = "strategy spec (M3: only 'single_device')" },
  { name = "precision", type = argrule.any, opt = true, default = "fp32",
    doc = "precision spec: 'fp32'|'amp_o1'|Precision instance" },
}(function(self, net, accelerator, strategy, precision)
  self.net = net
  self.accelerator = accel.Accelerator.create(accelerator)
  self.strategy = accel.Strategy.create(strategy)
  self.precision = accel.Precision.create(precision)
  self._optimizer = nil
  self._loss_fn = nil
  self._metrics = nil          -- MetricCollection
  self._trainer = nil
  self._current_lr = nil
end)

-- 前向:batch = {input, target}(或单 tensor)。Keras 用 net;Lightning 用
-- 子类 forward 覆写。返回 model(batch[1])。
function Model:forward(batch)
  local x
  if type(batch) == "table" then x = batch[1] else x = batch end
  return self.net(x)
end

-- 训练/推理态转发到 net(paddle.nn.Layer 有 train(mode)/eval())。
function Model:train(mode)
  if self.net and self.net.train then self.net:train(mode) end
  return self
end

function Model:eval()
  if self.net and self.net.eval then self.net:eval() end
  return self
end

-- ---------------------------------------------------------------------
-- Keras 侧 prepare/fit
-- ---------------------------------------------------------------------

-- prepare{optimizer, loss, metrics} -- Keras 装配。
Model.prepare = rule {
  { name = "self", type = argrule.any },
  { name = "optimizer", type = argrule.any, opt = true, default = nil,
    doc = "paddle.optimizer instance" },
  { name = "loss", type = argrule.any, opt = true, default = nil,
    doc = "loss callable: loss(pred, target) -> scalar tensor" },
  { name = "metrics", type = argrule.any, opt = true, default = nil,
    doc = "MetricCollection or {name=metric,...}" },
}(function(self, optimizer, loss, metrics)
  self._optimizer = optimizer
  self._loss_fn = loss
  if metrics ~= nil then
    if type(metrics) == "table" and metrics.log_dict ~= nil then
      self._metrics = metrics            -- 已是 MetricCollection
    else
      self._metrics = metrics_lib.MetricCollection(metrics)
    end
  end
end)

-- fit{loader, epochs} -- Keras 快捷训练(单优化器/单 loss)。
Model.fit = rule {
  { name = "self", type = argrule.any },
  { name = "loader", type = argrule.any,
    doc = "paddle.io.DataLoader" },
  { name = "epochs", type = argrule.number, opt = true, default = 1,
    doc = "epoch count" },
}(function(self, loader, epochs)
  local trainer = require("ocean.trainer").Trainer{
    max_epochs = epochs,
    accelerator = self.accelerator,
    strategy = self.strategy,
    precision = self.precision,
  }
  trainer:fit(self, loader)
end)

-- ---------------------------------------------------------------------
-- Lightning hooks(子类可覆写;默认 Keras 语义)
-- ---------------------------------------------------------------------

-- training_step(batch, batch_idx):返回 loss tensor。Keras 默认:net(x) 后
-- 用 _loss_fn(pred, target)。
function Model:training_step(batch, batch_idx)
  local pred = self:forward(batch)
  local target = batch[2]
  if self._loss_fn then
    return self._loss_fn(pred, target)
  end
  error("ocean.Model:training_step() not implemented (define it, or use "
        .. "Keras prepare{loss=...})", 2)
end

-- validation_step(batch, batch_idx):返回 None(或 loss)。Keras 默认:
-- net(x) + _loss_fn 算 val 指标。
function Model:validation_step(batch, batch_idx)
  local pred = self:forward(batch)
  local target = batch[2]
  if self._metrics then
    self._metrics:update(pred, target)
  end
  if self._loss_fn then
    return self._loss_fn(pred, target)
  end
  return nil
end

-- configure_optimizers():返回优化器实例。
function Model:configure_optimizers()
  if self._optimizer then return self._optimizer end
  error("ocean.Model:configure_optimizers() not implemented", 2)
end

-- configure_metrics():返回 MetricCollection(Keras prepare 已设则直接给)。
function Model:configure_metrics()
  return self._metrics
end

-- ---------------------------------------------------------------------
-- 辅助(供 Trainer / callbacks 用)
-- ---------------------------------------------------------------------

-- log(name, value):把标量记到当前 trainer 的 metrics 累积。
function Model:log(name, value, ...)
  if self._trainer then
    self._trainer:_model_log(name, value, ...)
  end
end

-- 当前学习率(取 optimizer 的 learning_rate;调度器后期接)。
function Model:current_lr()
  if self._optimizer then return self._optimizer.learning_rate end
  return self._current_lr
end

-- 参数统计
function Model:num_parameters()
  if self.net and self.net.parameters then
    local n = 0
    for _, p in self.net:named_parameters() do
      n = n + p:numel()
    end
    return n
  end
  return 0
end

function Model:num_trainable_parameters()
  return self:num_parameters()  -- M1 无 stop_gradient 区分,全算可训练
end

-- parameters():透传 net(供 optimizer 构造)。
function Model:parameters()
  if self.net and self.net.parameters then
    return self.net:parameters()
  end
  error("ocean.Model: net has no parameters()", 2)
end

-- state_dict():透传 net(供 save)。
function Model:state_dict()
  if self.net and self.net.state_dict then
    return self.net:state_dict()
  end
  error("ocean.Model: net has no state_dict()", 2)
end

-- set_state_dict(ckpt):从保存的字典恢复权重。
function Model:set_state_dict(d)
  -- net 的 load 接口:state_dict 键值 set 回参数(paddle-lua nn 有 set_* ?)
  if self.net and self.net.set_state_dict then
    self.net:set_state_dict(d)
    return
  end
  error("ocean.Model: net has no set_state_dict()", 2)
end

-- save(path):把 state_dict 逐 tensor 存盘。M1 paddle.save 只收单 tensor
-- (serialization.lua),这里每个参数存 `<path>.<name>`;load 逐文件读回。
-- 完整 .pdparams(pickle 表)随 paddle.save 的表支持落地后再切。
function Model:save(path)
  local d = self:state_dict()
  for name, t in pairs(d) do
    paddle.save{ obj = t, path = path .. "." .. name }
  end
end

-- load(path):从盘读回全部参数(state_dict 键与 save 时一致)。
function Model:load(path)
  local d = self:state_dict()
  local restored = {}
  for name in pairs(d) do
    restored[name] = paddle.load{ path = path .. "." .. name }
  end
  self:set_state_dict(restored)
end

return Model

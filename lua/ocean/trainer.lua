-- lua/ocean/trainer.lua -- Trainer:fit/validate/test/predict 驱动循环
--
-- Lightning 风格:负责训练循环、钩子调度、metrics 累积、callback 通知、
-- 日志转发。Keras 风格由 model.lua 的 Model:fit 内部构造本类实现。
--
-- 循环结构(每 epoch):
--   on_train_epoch_start -> 每 batch: on_train_batch_start ->
--     model:training_step(batch, idx) -> backward -> (grad_accum 时) step
--     -> on_train_batch_end -> 每 epoch end: on_train_epoch_end
--   (val_loader 时) on_validation_epoch_start -> 每 batch: validation_step
--     -> on_validation_epoch_end(callbacks 看 monitor)
-- 纯 Lua 5.1 + pl.class。

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"
local paddle = require "paddle"
local accel = require "ocean.accelerators"
local loggers_mod = require "ocean.loggers"
local callbacks_mod = require "ocean.callbacks"

local Trainer = class()

-- Trainer{max_epochs, accelerator, strategy, precision, callbacks, logger,
--         log_every_n_steps, enable_checkpointing, enable_progress_bar}
Trainer._init = rule {
  { name = "self", type = argrule.any },
  { name = "max_epochs", type = argrule.number, opt = true, default = 1,
    doc = "epoch count" },
  { name = "accelerator", type = argrule.any, opt = true, default = "cpu",
    doc = "'cpu'|'cuda'|Accelerator instance" },
  { name = "strategy", type = argrule.any, opt = true, default = "single_device",
    doc = "'single_device'|Strategy instance" },
  { name = "precision", type = argrule.any, opt = true, default = "fp32",
    doc = "'fp32'|'amp_o1'|Precision instance" },
  { name = "callbacks", type = argrule.any, opt = true, default = nil,
    doc = "{Callback,...} array" },
  { name = "logger", type = argrule.any, opt = true, default = nil,
    doc = "Logger instance; omit for MemoryLogger" },
  { name = "log_every_n_steps", type = argrule.number, opt = true, default = 10,
    doc = "log metrics every N steps" },
  { name = "enable_checkpointing", type = argrule.boolean, opt = true,
    default = true, doc = "auto add ModelCheckpoint (val_loss monitor)" },
  { name = "enable_progress_bar", type = argrule.boolean, opt = true,
    default = true, doc = "auto add ProgressBar" },
}(function(self, max_epochs, accelerator, strategy, precision,
           callbacks, logger, log_every_n_steps,
           enable_checkpointing, enable_progress_bar)
  self.max_epochs = max_epochs
  self.accelerator = accel.Accelerator.create(accelerator)
  self.strategy = accel.Strategy.create(strategy)
  self.precision = accel.Precision.create(precision)
  self.log_every_n_steps = log_every_n_steps
  self._logger = logger or loggers_mod.MemoryLogger()
  self._callbacks = {}
  if callbacks then
    for _, cb in ipairs(callbacks) do
      self:add_callback(cb)
    end
  end
  if enable_checkpointing then
    self:add_callback(callbacks_mod.ModelCheckpoint{ monitor = "val_loss" })
  end
  if enable_progress_bar then
    self:add_callback(callbacks_mod.ProgressBar{})
  end
  -- 运行态
  self._model = nil
  self._step = 0              -- 全局 step
  self._epoch = 0
  self._max_steps = 0
  self._should_stop = false
  self._grad_accum = 1        -- GradAccum 设置的累积因子
  self._metrics_acc = {}      -- 当前 epoch 累积的指标 {name=last_value}
  self._fit_log = self._logger
end)

-- add_callback(cb):注册回调(去重按实例)。
function Trainer:add_callback(cb)
  if type(cb) == "table" then
    self._callbacks[#self._callbacks + 1] = cb
  end
end

-- ---------------------------------------------------------------------
-- 回调通知(按固定顺序)
-- ---------------------------------------------------------------------

function Trainer:_fire(hook, ...)
  for _, cb in ipairs(self._callbacks) do
    local f = cb[hook]
    if f then f(cb, self, ...) end
  end
end

-- ---------------------------------------------------------------------
-- 供回调/model 查询
-- ---------------------------------------------------------------------

function Trainer:max_steps() return self._max_steps end
function Trainer:current_step() return self._step end
function Trainer:current_epoch() return self._epoch end
function Trainer:should_stop() return self._should_stop end
function Trainer:stop_training() self._should_stop = true end
function Trainer:_set_grad_accum(n) self._grad_accum = n end

-- monitor_value(name):当前累积指标里的值(供 EarlyStopping/Checkpoint)。
function Trainer:monitor_value(name)
  return self._metrics_acc[name]
end

-- log_metrics(map, step, epoch):把指标表交给 logger + 本地累积。
function Trainer:log_metrics(map, step, epoch)
  for k, v in pairs(map) do
    self._metrics_acc[k] = loggers_mod.MetricValue.to_number(v)
  end
  self._logger:log(map, step, epoch)
end

-- Model:log(name, value) 的落地:交给当前 metrics 累积 + logger。
function Trainer:_model_log(name, value)
  self:log_metrics({ [name] = value }, self._step, self._epoch)
end

-- ---------------------------------------------------------------------
-- 主入口
-- ---------------------------------------------------------------------

-- fit(model, train_loader, val_loader):训练 + (可选)验证。
function Trainer:fit(model, train_loader, val_loader)
  self._model = model
  model._trainer = self
  self._epoch = 0
  self._should_stop = false
  -- 优化器装配(子类 configure_optimizers;Keras 已 prepare)
  model._optimizer = model:configure_optimizers()

  self:_fire("on_fit_start", model)

  local n_epochs = self.max_epochs
  for e = 1, n_epochs do
    if self._should_stop then break end
    self._epoch = e
    self:_run_epoch(model, train_loader, val_loader)
  end

  self:_fire("on_fit_end", model)
  self._logger:close()
  model._trainer = nil
  return self
end

-- _run_epoch:单 epoch 训练 + 验证。
function Trainer:_run_epoch(model, train_loader, val_loader)
  self._metrics_acc = {}
  self:_fire("on_train_epoch_start", model, self._epoch)

  -- 预估总步数(供 ProgressBar)
  local n_batches = 0
  if train_loader and train_loader.len then n_batches = train_loader:len() end
  self._max_steps = self._max_steps + n_batches

  local batch_idx = 0
  for _, batch in train_loader:iter() do
    if self._should_stop then break end
    self._step = self._step + 1
    batch_idx = batch_idx + 1
    self:_fire("on_train_batch_start", model, batch, batch_idx)
    model:train()   -- 训练态

    local loss = model:training_step(batch, batch_idx)
    -- backward + step
    if loss ~= nil then
      model._optimizer:clear_grad()
      paddle.autograd.backward{ loss }
      if batch_idx % self._grad_accum == 0 then
        model._optimizer:step()
      end
    end

    self:_fire("on_train_batch_end", model, loss, batch, batch_idx)
    if batch_idx % self.log_every_n_steps == 0 then
      self._logger:log(self._metrics_acc, self._step, self._epoch)
    end
  end

  self:_fire("on_train_epoch_end", model, self._epoch)

  -- 验证
  if val_loader then
    self:_run_validation(model, val_loader)
  end
end

-- _run_validation:验证 epoch(不训练,不更新权重)。
function Trainer:_run_validation(model, val_loader)
  self:_fire("on_validation_epoch_start", model, self._epoch)
  model:eval()
  local batch_idx = 0
  for _, batch in val_loader:iter() do
    batch_idx = batch_idx + 1
    self:_fire("on_validation_batch_start", model, batch, batch_idx)
    model:validation_step(batch, batch_idx)
    self:_fire("on_validation_batch_end", model, nil, batch, batch_idx)
  end
  -- Keras 装配的 MetricCollection:compute 各指标并写日志
  -- (torchmetrics/Keras 语义:验证结果每 epoch 上报;reset 留到 epoch 边界)
  local mc = type(model.configure_metrics) == "function"
                and model:configure_metrics()
  if mc ~= nil and type(mc.log_dict) == "function" then
    local log = mc:log_dict()
    for name, v in pairs(log) do
      self._metrics_acc["val_" .. name] =
        loggers_mod.MetricValue.to_number(v)
    end
    self._logger:log(self._metrics_acc, self._step, self._epoch)
    mc:reset()
  end
  self:_fire("on_validation_epoch_end", model, self._epoch)
end

-- validate(model, val_loader):单独验证(返回 log 表)。
function Trainer:validate(model, val_loader)
  model._trainer = self
  self._model = model
  self._metrics_acc = {}
  self:_run_validation(model, val_loader)
  model._trainer = nil
  return self._metrics_acc
end

-- test(model, test_loader):测试(只算 metrics)。
function Trainer:test(model, test_loader)
  model._trainer = self
  self._model = model
  self._metrics_acc = {}
  self:_fire("on_test_epoch_start", model)
  model:eval()
  local batch_idx = 0
  for _, batch in test_loader:iter() do
    batch_idx = batch_idx + 1
    local pred = model:forward(batch)
    if model._metrics then model._metrics:update(pred, batch[2]) end
  end
  self:_fire("on_test_epoch_end", model)
  model._trainer = nil
  return self._metrics_acc
end

-- predict(model, loader):逐 batch 前向,收集输出。
function Trainer:predict(model, loader)
  model._trainer = self
  self._model = model
  local out = {}
  if type(model.eval) == "function" then model:eval() end
  local i = 0
  for _, batch in loader:iter() do
    i = i + 1
    out[i] = model:forward(batch)
  end
  model._trainer = nil
  return out
end

return Trainer

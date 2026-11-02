-- lua/ocean/callbacks.lua -- Callback 基类 + M3 前 8 个实现
--
-- Lightning/Ocean Callback 语义:Trainer 在训练循环的关键节点调用
-- 回调钩子。基类定义全部钩子为 no-op,子类只覆写需要的。
-- 钩子签名(与 Ocean 对齐):
--   on_fit_start(trainer, model)
--   on_train_epoch_start(trainer, model, epoch)
--   on_train_batch_start(trainer, model, batch, batch_idx)
--   on_train_batch_end(trainer, model, outputs, batch, batch_idx)
--   on_train_epoch_end(trainer, model, epoch)
--   on_validation_epoch_start / end
--   on_fit_end(trainer, model)
-- M3 八个(overview.md §6.3):
--   Checkpoint / EarlyStopping / LRMonitor / ProgressBar / Timer /
--   GradAccum / ModelSummary / RichLog

local class = require "pl.class"

local Callback = class()

local function noop() end

local hooks = {
  "on_fit_start", "on_fit_end",
  "on_train_epoch_start", "on_train_epoch_end",
  "on_train_batch_start", "on_train_batch_end",
  "on_validation_epoch_start", "on_validation_epoch_end",
  "on_validation_batch_start", "on_validation_batch_end",
  "on_test_epoch_start", "on_test_epoch_end",
  "on_predict_batch_start", "on_predict_batch_end",
}
for _, h in ipairs(hooks) do
  if Callback[h] == nil then Callback[h] = noop end
end

-- ---------------------------------------------------------------------
-- 1. ModelCheckpoint:监控 val_loss 等指标,最优时保存模型。
--    保存走 model:save(path)(paddle-lua save 接口)。
-- ---------------------------------------------------------------------

local ModelCheckpoint = class(Callback)

function ModelCheckpoint:_init(config)
  config = config or {}
  self.monitor = config.monitor or "val_loss"
  self.mode = config.mode or "min"        -- min|max
  self.dirpath = config.dirpath or "checkpoints"
  self.filename = config.filename or "best"
  self.save_top_k = config.save_top_k or 1
  self._best = nil
  self._best_epoch = nil
end

function ModelCheckpoint:on_validation_epoch_end(trainer, model, epoch)
  local v = trainer:monitor_value(self.monitor)
  if v == nil then return end
  local better
  if self._best == nil then
    better = true
  elseif self.mode == "min" then
    better = v < self._best
  else
    better = v > self._best
  end
  if better then
    self._best = v
    self._best_epoch = epoch
    local p = self.dirpath .. "/" .. self.filename .. ".pdparams"
    model:save(p)
    if trainer._logger and trainer._logger._print then
      print("ModelCheckpoint: saved best (" .. self.monitor .. "="
            .. ("%.6g"):format(v) .. ", epoch " .. tostring(epoch) .. ")")
    end
  end
end

-- ---------------------------------------------------------------------
-- 2. EarlyStopping:监控指标连续 patience 轮不改善则停止。
-- ---------------------------------------------------------------------

local EarlyStopping = class(Callback)

function EarlyStopping:_init(config)
  config = config or {}
  self.monitor = config.monitor or "val_loss"
  self.mode = config.mode or "min"
  self.patience = config.patience or 5
  self.min_delta = config.min_delta or 0.0
  self._bad_epochs = 0
  self._best = nil
  self._stopped = false
end

function EarlyStopping:on_validation_epoch_end(trainer, model, epoch)
  local v = trainer:monitor_value(self.monitor)
  if v == nil then return end
  local improved
  if self._best == nil then
    improved = true
  elseif self.mode == "min" then
    improved = v < self._best - self.min_delta
  else
    improved = v > self._best + self.min_delta
  end
  if improved then
    self._best = v
    self._bad_epochs = 0
  else
    self._bad_epochs = self._bad_epochs + 1
    if self._bad_epochs >= self.patience then
      self._stopped = true
      trainer:stop_training()
      print("EarlyStopping: no improvement for " .. self.patience
            .. " epochs, stopping")
    end
  end
end

-- ---------------------------------------------------------------------
-- 3. LRMonitor:每个 epoch 记录当前 learning rate 到日志。
-- ---------------------------------------------------------------------

local LRMonitor = class(Callback)

function LRMonitor:_init(config)
  config = config or {}
  self.logging_interval = config.logging_interval or "epoch"  -- epoch|step
end

function LRMonitor:on_train_epoch_end(trainer, model, epoch)
  local lr = model:current_lr()
  if lr ~= nil then
    trainer:log_metrics({ lr = lr }, 0, epoch)
  end
end

-- ---------------------------------------------------------------------
-- 4. ProgressBar:打印训练进度(每 batch 更新,5.1 无 \r 终端控制,
--    用简单每 10% 输出一行的轻量方案)。
-- ---------------------------------------------------------------------

local ProgressBar = class(Callback)

function ProgressBar:_init(config)
  config = config or {}
  self.refresh_every = config.refresh_every or 0.1
  self._next = 0
end

function ProgressBar:on_train_batch_end(trainer, model, outputs, batch, batch_idx)
  if trainer:max_steps() == 0 then return end
  local frac = trainer:current_step() / trainer:max_steps()
  if frac >= self._next then
    print(("Training: %d%%"):format(math.floor(frac * 100)))
    self._next = self._next + self.refresh_every
  end
end

-- ---------------------------------------------------------------------
-- 5. Timer:记录 fit 总时长(秒)。
-- ---------------------------------------------------------------------

local Timer = class(Callback)

function Timer:_init(config)
  config = config or {}
  self._t0 = nil
  self.elapsed = nil
end

function Timer:on_fit_start(trainer, model)
  self._t0 = os.clock()
end

function Timer:on_fit_end(trainer, model)
  if self._t0 then
    self.elapsed = os.clock() - self._t0
    if trainer._logger and trainer._logger._print then
      print(("Timer: fit took %.2fs"):format(self.elapsed))
    end
  end
end

-- ---------------------------------------------------------------------
-- 6. GradAccum:梯度累积(accumulate 个 batch 才 step 一次)。
--    在 on_train_batch_end 里触发 optimizer:step 的合并 —— 这里简化为
--    标记,真正的 step 调度在 Trainer 里看 _grad_accum_batch。
-- ---------------------------------------------------------------------

local GradAccum = class(Callback)

function GradAccum:_init(config)
  config = config or {}
  self.accumulate = config.accumulate or 1   -- 每 N 个 batch 更新一次
  self._count = 0
end

function GradAccum:on_train_batch_end(trainer, model, outputs, batch, batch_idx)
  self._count = self._count + 1
  trainer:_set_grad_accum(self.accumulate)
end

function GradAccum:should_step()
  return self._count % self.accumulate == 0
end

-- ---------------------------------------------------------------------
-- 7. ModelSummary:fit 开始时打印模型参数统计。
-- ---------------------------------------------------------------------

local ModelSummary = class(Callback)

function ModelSummary:_init(config)
  config = config or {}
end

function ModelSummary:on_fit_start(trainer, model)
  local n_params = model:num_parameters()
  local n_trainable = model:num_trainable_parameters()
  print(("ModelSummary: %d parameters (%d trainable)"):format(n_params, n_trainable))
end

-- ---------------------------------------------------------------------
-- 8. RichLog:把指标以富文本形式记录到 logger(Rich 的 Lua 降级 = 直接
--    调用 logger:log)。
-- ---------------------------------------------------------------------

local RichLog = class(Callback)

function RichLog:_init(config)
  config = config or {}
end

function RichLog:on_train_epoch_end(trainer, model, epoch)
  -- 由 Trainer 负责把当前 metrics 交给 logger;这里只做提示。
end

return {
  Callback = Callback,
  ModelCheckpoint = ModelCheckpoint,
  EarlyStopping = EarlyStopping,
  LRMonitor = LRMonitor,
  ProgressBar = ProgressBar,
  Timer = Timer,
  GradAccum = GradAccum,
  ModelSummary = ModelSummary,
  RichLog = RichLog,
}

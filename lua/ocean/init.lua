-- lua/ocean/init.lua -- ocean 入口
-- Lightning/Keras 风格训练框架(paddle-lua 生态,纯 Lua 零编译)。
--   local ocean = require "ocean"
--   -- ① Lightning:子类 ocean.Model 定义 hooks,Trainer 驱动
--   -- ② Keras:Model{net=...}:prepare{}:fit{}
-- 设计:paddle-lua docs/migration/plan/overview.md §6.3。

local M = {}

local accel = require "ocean.accelerators"
M.Accelerator = accel.Accelerator
M.CPUAccelerator = accel.CPUAccelerator
M.CUDAAccelerator = accel.CUDAAccelerator
M.Strategy = accel.Strategy
M.SingleDeviceStrategy = accel.SingleDeviceStrategy
M.Precision = accel.Precision
M.FP32Precision = accel.FP32Precision
M.AMPO1Precision = accel.AMPO1Precision

M.Model = require "ocean.model"
M.Trainer = require "ocean.trainer"
M.DataModule = require "ocean.datamodule"

local loggers = require "ocean.loggers"
M.Logger = loggers.Logger
M.MemoryLogger = loggers.MemoryLogger
M.CSVLogger = loggers.CSVLogger
M.ConsoleLogger = loggers.ConsoleLogger

local cbs = require "ocean.callbacks"
M.Callback = cbs.Callback
M.callbacks = {
  ModelCheckpoint = cbs.ModelCheckpoint,
  EarlyStopping = cbs.EarlyStopping,
  LRMonitor = cbs.LRMonitor,
  ProgressBar = cbs.ProgressBar,
  Timer = cbs.Timer,
  GradAccum = cbs.GradAccum,
  ModelSummary = cbs.ModelSummary,
  RichLog = cbs.RichLog,
}

return M
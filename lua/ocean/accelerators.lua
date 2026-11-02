-- lua/ocean/accelerators.lua -- Accelerator / Strategy / Precision 三层分离
--
-- Ocean 的"平台层":Accelerator(设备)/ Strategy(分布式,只做 SingleDevice)/
-- Precision(精度)。三层正交,Model/Trainer 通过它们决定怎么跑。
-- M3 范围(overview.md §6.3):
--   * Accelerators: CPU / CUDA(其余靠 paddle.device 泛化)
--   * Strategies:   SingleDevice(分布式测不了,已延后)
--   * Precision:    fp32 + AMP O1
-- 纯 Lua 5.1 + pl.class,全部走 paddle-lua 公开 API。

local class = require "pl.class"

-- ---------------------------------------------------------------------
-- Accelerator:设备抽象。CPU 与 CUDA 差异在 paddle-lua 目前是纯 CPU
-- 构建,place 用 CPU;CUDA 占位(device 语义保留,无 GPU 环境不展开)。
-- ---------------------------------------------------------------------

local Accelerator = class()

function Accelerator:_init(config)
  config = config or {}
  self.type = config.type or "cpu"
end

-- device: 返回该加速器对应的 place(供 tensor 创建/移动用)。
function Accelerator:device()
  return "cpu"  -- 纯 CPU 构建;CUDA 在 paddle-lua 有 GPU 时再泛化
end

-- 是否支持某个特性(AMP 等)。
function Accelerator:is_available()
  return true
end

local CPUAccelerator = class(Accelerator)
function CPUAccelerator:_init(config)
  Accelerator._init(self, config)
  self.type = "cpu"
end

local CUDAAccelerator = class(Accelerator)
function CUDAAccelerator:_init(config)
  Accelerator._init(self, config)
  self.type = "cuda"
end
function CUDAAccelerator:is_available()
  -- 无 GPU 环境;有 paddle.device.is_compiled_with_cuda 时再探测
  return false
end

-- 工厂:accelerator = "cpu"|"cuda" 或实例。
-- 实例判据:表 + 有 is_available 方法(pl.class 实例继承类方法)。
function Accelerator.create(spec)
  if type(spec) == "table" and type(spec.is_available) == "function" then
    return spec
  end
  spec = spec or "cpu"
  if spec == "cpu" then return CPUAccelerator() end
  if spec == "cuda" then return CUDAAccelerator() end
  error("ocean: unknown accelerator '" .. tostring(spec) .. "'", 3)
end

-- ---------------------------------------------------------------------
-- Strategy:并行策略。M3 只做 SingleDevice(单卡/单进程)。
-- ---------------------------------------------------------------------

local Strategy = class()
function Strategy:_init(config)
  config = config or {}
  self.type = config.type or "single_device"
end
function Strategy:is_distributed()
  return false
end

local SingleDeviceStrategy = class(Strategy)
function SingleDeviceStrategy:_init(config)
  Strategy._init(self, config)
  self.type = "single_device"
end

function Strategy.create(spec)
  if type(spec) == "table" and type(spec.is_distributed) == "function" then
    return spec
  end
  spec = spec or "single_device"
  if spec == "single_device" then return SingleDeviceStrategy() end
  error("ocean: unknown strategy '" .. tostring(spec) .. "' (only SingleDevice in M3)", 3)
end

-- ---------------------------------------------------------------------
-- Precision:fp32 / AMP O1(混合精度)。
-- ---------------------------------------------------------------------

local Precision = class()
function Precision:_init(config)
  config = config or {}
  self.type = config.type or "fp32"
end

local FP32Precision = class(Precision)
function FP32Precision:_init(config)
  Precision._init(self, config)
  self.type = "fp32"
end

local AMPO1Precision = class(Precision)
function AMPO1Precision:_init(config)
  Precision._init(self, config)
  self.type = "amp_o1"
end

function Precision.create(spec)
  if type(spec) == "table" and type(spec.type) == "string" then
    return spec
  end
  spec = spec or "fp32"
  if spec == "fp32" then return FP32Precision() end
  if spec == "amp_o1" or spec == "amp" then return AMPO1Precision() end
  error("ocean: unknown precision '" .. tostring(spec) .. "' (fp32 / amp_o1)", 3)
end

-- 导出:加速器/策略/精度 三类。

return {
  Accelerator = Accelerator,
  CPUAccelerator = CPUAccelerator,
  CUDAAccelerator = CUDAAccelerator,
  Strategy = Strategy,
  SingleDeviceStrategy = SingleDeviceStrategy,
  Precision = Precision,
  FP32Precision = FP32Precision,
  AMPO1Precision = AMPO1Precision,
}

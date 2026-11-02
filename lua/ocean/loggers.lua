-- lua/ocean/loggers.lua -- Logger 抽象 + M3 前 3 个实现
--
-- Lightning/Ocean Logger 语义:trainer 在 step/epoch 结束时调用
-- logger:log(metrics_map, step, epoch),logger 决定怎么落盘。
-- M3 三个(overview.md §6.3):
--   * MemoryLogger:内存表累积(可查询/可后续转存)← 默认
--   * CSVLogger:CSV 文件,每 epoch 追加一行(纯 Lua 手写,零依赖)
--   * ConsoleLogger:直接 print(ProgressBar 的文本伴侣)
-- VisualDL 需 paddle-lua 的 VisualDL 绑定,暂不做。

local class = require "pl.class"

-- ---------------------------------------------------------------------
-- 值归一:MetricValue 把 Tensor/数字 归一成 Lua number。
-- 兼容 metrics-lua 的标量读取(0-d/1 元素 cpu tensor 的 f32 LE 首 4 字节)。
-- ---------------------------------------------------------------------

local MetricValue = {}

function MetricValue.to_number(v)
  if type(v) == "number" then return v end
  if type(v) == "userdata" then
    local ok, s = pcall(function() return v:to_host_bytes() end)
    if not ok then return 0 end          -- GPU/不可读:占位 0
    if s == nil then return 0 end
    local b1, b2, b3, b4 = s:byte(1, 4)
    if not b1 then return 0 end          -- 空 tensor
    local bits = b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
    local sign = bits >= 2147483648 and 1 or 0
    local exp = math.floor(bits / 8388608) % 256
    local mant = bits % 8388608
    if exp == 0 then return 0.0 end
    if exp == 255 then return sign == 1 and -math.huge or math.huge end
    local val = (1 + mant / 8388608) * 2 ^ (exp - 127)
    return sign == 1 and -val or val
  end
  return tonumber(v) or 0
end

-- ---------------------------------------------------------------------

local Logger = class()

function Logger:_init(config)
  config = config or {}
  self.name = config.name or "logger"
end

-- log(metrics, step, epoch):metrics 是 {name=value,...},value 可以是
-- Lua number 或 paddle Tensor(取标量)。step 是 batch 序号,epoch 是轮次。
function Logger:log(metrics, step, epoch)
  error("ocean: Logger:log() must be implemented", 2)
end

function Logger:close()
  -- 默认 no-op;CSV 等在 close 时 flush
end

-- ---------------------------------------------------------------------

-- MemoryLogger:最简单也最有用 —— 调试/测试直接读表断言。
local MemoryLogger = class(Logger)

function MemoryLogger:_init(config)
  Logger._init(self, config or { name = "memory" })
  self.epochs = {}   -- epochs[e] = { name = {step=value, ...} }
  self.steps = {}    -- 最近一次 step 的完整快照
  self._cur_epoch = nil
  self._cur = nil
end

function MemoryLogger:log(metrics, step, epoch)
  if epoch ~= nil and epoch ~= self._cur_epoch then
    self._cur_epoch = epoch
    self.epochs[epoch] = self.epochs[epoch] or {}
    self._cur = self.epochs[epoch]
  end
  local target = self._cur or self.steps
  for name, v in pairs(metrics) do
    target[name] = target[name] or {}
    target[name][step or 0] = MetricValue.to_number(v)
  end
end

-- epoch_metric(name, epoch):该 epoch 最后一次记录的值。
function MemoryLogger:epoch_metric(name, epoch)
  local e = self.epochs[epoch]
  if not e or not e[name] then return nil end
  local last
  for _, v in pairs(e[name]) do last = v end
  return last
end

-- 所有 epoch 该 name 的末值表(供 EarlyStopping 用)。
function MemoryLogger:history(name)
  local out = {}
  for e = 1, #self.epochs do
    local v = self:epoch_metric(name, e)
    if v ~= nil then out[e] = v end
  end
  return out
end

-- ---------------------------------------------------------------------

-- CSVLogger:每 epoch 一行的 CSV(header 为指标名,确定性排序)。
-- 纯 Lua 5.1 手写:io.open 直接写,数字/字符串转义。

local CSVLogger = class(Logger)

function CSVLogger:_init(config)
  Logger._init(self, config or { name = "csv" })
  self.filename = config.filename or "ocean_log.csv"
  self._header_written = false
  self._names = {}      -- 已见过的指标名(顺序)
  for i, v in ipairs(config.columns or {}) do
    self._names[i] = v
  end
  self._fg = io.open(self.filename, "w")
  if not self._fg then
    error("ocean: CSVLogger cannot open '" .. self.filename .. "'", 2)
  end
end

function CSVLogger:_column(name)
  for i, n in ipairs(self._names) do
    if n == name then return i end
  end
  self._names[#self._names + 1] = name
  return #self._names
end

function CSVLogger:log(metrics, step, epoch)
  if epoch == nil then return end
  if not self._header_written then
    -- 确定性列序:先 config.columns(若有),再按名称字母序补全
    local names = {}
    for name in pairs(metrics) do
      names[#names + 1] = name
    end
    table.sort(names)
    for _, name in ipairs(names) do self:_column(name) end
    self._header_written = true
  end
  local row = {}
  for i, name in ipairs(self._names) do
    local v = metrics[name]
    if v == nil then
      row[i] = ""
    else
      row[i] = ("%.6g"):format(MetricValue.to_number(v))
    end
  end
  self._fg:write(table.concat(row, ","), "\n")
  self._fg:flush()
end

function CSVLogger:close()
  if not self._fg then return end
  self._fg:close()
  self._fg = nil
  -- header 写到最前:读回正文再重写
  local f = io.open(self.filename, "r")
  local body
  if f then body = f:read("*a"); f:close() else body = "" end
  f = io.open(self.filename, "w")
  f:write(table.concat(self._names, ","), "\n", body)
  f:close()
end

-- ---------------------------------------------------------------------

-- ConsoleLogger:print 每次 log(ProgressBar 的文本伴侣)。
local ConsoleLogger = class(Logger)

function ConsoleLogger:_init(config)
  Logger._init(self, config or { name = "console" })
end

function ConsoleLogger:log(metrics, step, epoch)
  local parts = {}
  for name, v in pairs(metrics) do
    parts[#parts + 1] = name .. "=" .. ("%.4g"):format(MetricValue.to_number(v))
  end
  local head = epoch and ("[epoch %d step %d]"):format(epoch, step or 0)
                 or ("[step %d]"):format(step or 0)
  print(head, table.concat(parts, " "))
end

return {
  Logger = Logger,
  MemoryLogger = MemoryLogger,
  CSVLogger = CSVLogger,
  ConsoleLogger = ConsoleLogger,
  MetricValue = MetricValue,
}

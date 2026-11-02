-- lua/ocean/datamodule.lua -- DataModule:数据集/DataLoader 组装(薄)
--
-- Lightning DataModule 语义:把 train/val/test DataLoader 集中到一个
-- 对象,predict/validate/fit 复用同一实例。
--   local dm = ocean.DataModule{ train = train_loader, val = val_loader,
--                                test = test_loader }
--   trainer:fit(model, dm.train, dm.val)

local class = require "pl.class"
local argrule = require "argrule"
local rule = require "argrule.rule"

local DataModule = class()

DataModule._init = rule {
  { name = "self", type = argrule.any },
  { name = "train", type = argrule.any, opt = true, default = nil,
    doc = "train DataLoader (or nil)" },
  { name = "val", type = argrule.any, opt = true, default = nil,
    doc = "validation DataLoader (or nil)" },
  { name = "test", type = argrule.any, opt = true, default = nil,
    doc = "test DataLoader (or nil)" },
  { name = "predict", type = argrule.any, opt = true, default = nil,
    doc = "predict DataLoader (or nil)" },
}(function(self, train, val, test, predict)
  self.train = train
  self.val = val
  self.test = test
  self.predict = predict
end)

-- setup():用户子类可覆写,在这里构造 DataLoader(延迟加载)。
function DataModule:setup()
  -- default no-op;子类在这里 build train/val/test
end

return DataModule
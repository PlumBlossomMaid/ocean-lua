package = "ocean"
version = "scm-1"

source = {
    url = "git+https://github.com/PlumBlossomMaid/ocean-lua.git",
    branch = "main",
}

description = {
    summary = "Lightning/Keras-style trainer for paddle-lua",
    detailed = [[
        ocean-lua is a pure-Lua, zero-compile training framework on top of
        paddle-lua's public API. Dual-mode: Lightning-style hooks driven by
        Trainer:fit, and Keras-style Model:prepare/fit. Three-layer
        Accelerator/Strategy/Precision split (CPU/CUDA, SingleDevice, fp32/
        AMP O1). First batch: Model + Trainer fit/validate/test/predict +
        DataModule + 8 callbacks + 3 loggers.

        Ecosystem constraint (paddle-lua overview.md sec.1): only calls the
        public API of paddle-lua and metrics-lua.
    ]],
    homepage = "https://github.com/PlumBlossomMaid/ocean-lua",
    license = "Apache-2.0",
}

-- Hard deps: Penlight (pl.class/pl.List, the shared ecosystem base --
-- paddle-lua D34: NEVER vendor it) and argrule (argument parsing, D31).
-- paddle-lua computes on Tensors; metrics-lua supplies MetricCollection
-- for log_dict integration.
dependencies = {
    "lua >= 5.1",
    "penlight >= 1.13",
    "argrule >= 0.1, < 0.2",
    "paddle-lua >= 0.1",
    "metrics-lua >= 0.1",
}

build = {
    type = "builtin",
    modules = {
        ["ocean"] = "lua/ocean/init.lua",
        ["ocean.model"] = "lua/ocean/model.lua",
        ["ocean.trainer"] = "lua/ocean/trainer.lua",
        ["ocean.datamodule"] = "lua/ocean/datamodule.lua",
        ["ocean.callbacks"] = "lua/ocean/callbacks.lua",
        ["ocean.loggers"] = "lua/ocean/loggers.lua",
        ["ocean.accelerators"] = "lua/ocean/accelerators.lua",
    },
}
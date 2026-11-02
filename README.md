# ocean-lua

Lightning/Keras-style training framework for [paddle-lua](https://github.com/PlumBlossomMaid/Paddle-Lua) —
pure Lua 5.1, zero native compile, three-layer Accelerator/Strategy/Precision split.

```lua
local ocean = require "ocean"

-- ① Lightning style: define hooks, Trainer drives the loop
local MyModel = class(ocean.Model)
function MyModel:init()
  MyModel.super.init(self)
  self.net = paddle.nn.Linear(28, 10)
end
function MyModel:training_step(batch, batch_idx)
  local loss = paddle.nn.functional.cross_entropy(self.net(batch[1]), batch[2])
  self:log("train_loss", loss)
  return loss
end
function MyModel:configure_optimizers()
  return paddle.optimizer.Adam{ learning_rate = 1e-3, parameters = self:parameters() }
end
local trainer = ocean.Trainer{ max_epochs = 10, accelerator = "cpu" }
trainer:fit(model, train_loader, val_loader)

-- ② Keras style: quick prototype
local model = ocean.Model{ net = paddle.nn.Sequential(...) }
model:prepare{ optimizer = opt, loss = loss_fn, metrics = { acc } }
model:fit{ train_loader, epochs = 10 }
```

## M3 scope (subset of Ocean's ~140 source files)

| component | Lua M3 scope |
|---|---|
| `Model` (dual-mode) | full |
| `Trainer` fit/validate/test/predict | full |
| `DataModule` | thin |
| Callbacks (18) | first 8: Checkpoint / EarlyStopping / LRMonitor / ProgressBar / Timer / GradAccum / ModelSummary / RichLog |
| Loggers (9) | first 3: CSV / in-memory / (VisualDL pending paddle-lua binding) |
| Strategies (6) | SingleDevice only (distributed untestable, deferred) |
| Accelerators (7) | CPU / CUDA |
| Precision (4) | fp32 + AMP O1 |

Anything depending on "Python ecosystem integration" (wandb/mlflow/comet/
deepspeed) is out — those libraries have no Lua client.

## Design

Consumes only paddle-lua's **public API** (`paddle.nn.*`, `paddle.optimizer.*`,
`paddle.io.*`, metrics-lua `MetricCollection:log_dict()`) — ecosystem repos
must not reach into the C ABI (paddle-lua overview.md §1).

## Dependencies (rocks)

- `penlight`(pl.class, shared ecosystem base — never vendored)
- `argrule`(argument parsing)
- `paddle-lua`(runtime)
- `metrics-lua`(runtime, for log_dict integration)

## Run the smoke

From the paddle-lua repo root (so `./build-test/paddle_core.dll`
resolves) with the usual paddle runtime PATH:

```bash
luajit E:\code\ocean-lua\tools\smoke\ocean_smoke.lua
# SMOKE PASS: ocean-lua first batch OK (N assertions)
```

## License

Apache-2.0.

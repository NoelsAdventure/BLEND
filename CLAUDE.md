# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

BLEND / GenSafeNav: PPO-Lagrangian crowd-navigation policies trained with conformal-inference-augmented observations, plus a LoRA-based adaptation layer used to switch behaviour online when humans become unaware of the robot. The paper is "Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling" (CoRL 2025) — arXiv: https://arxiv.org/abs/2508.05634v1. ROS 2 deployment lives in a separate repo: https://github.com/tasl-lab/GenSafeNav-ROS2. The repo here is research code — train.py / test.py / visualize.py are the entrypoints, everything else is library code or analysis.

## Environment

The intended runtime is the Docker image built from `Dockerfile` (PyTorch 2.3.1 + CUDA 12.1 + Python-RVO2 compiled from the bundled source). `Python-RVO2` must be built before anything else runs; the Dockerfile does this via `python setup.py build && python setup.py install`. Outside Docker, build it manually from the `Python-RVO2/` directory.

```bash
# Build (USER_ID/GROUP_ID args keep mounted volumes writable from the host)
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t gen_safe_py10:latest .
# Run (port 12345 is for Jupyter; /workspace is the repo mount)
docker run --runtime=nvidia -it -p 12345:8888 -v /home/docker_share:/home/ -v $(pwd):/workspace gen_safe_py10:latest /bin/bash
```

## Common commands

```bash
# Train a new policy (config lives in crowd_nav/configs/config.py + arguments.py)
python train.py

# Test a pretrained policy
python test.py --model_dir trained_models/<NAME>/ --test_model <CHECKPOINT>.pt

# Render trajectories / save GIFs
python visualize.py --model_dir trained_models/<NAME>/ --test_model <CHECKPOINT>.pt --save_slides

# Full adaptive-LoRA proof-of-concept sweep (4 scenarios × N behaviours), then aggregate + plot
./test_adaptive_lora_poc.sh

# Train the friendly/aware-human predictor used by the `*_pred` LoRA behaviours
python train_alpha_predictor.py
```

Other sweep / evaluation shell scripts (each hardcodes a `MODEL_DIR` + `CHECKPOINT` near the top — edit those to retarget):
- `test_lorab_invisible.sh` — single test run for the invisible-robot fulltune baseline.
- `test_lorab_gradual.sh` — fixed-scale test (currently scale=1.0; the script has a commented-out 0.0→2.0 sweep).
- `test_lorae_scenarios.sh` — invisible-scenario test for the LoraE adapter with `--lora_scale 0.0` (LoRA disabled). A visible-scenario block is commented out alongside it.
- `test_awareness_accuracy.sh` — sweeps `--discrepancy_threshold` against the `seperate_mixed_5050` scenario to evaluate the discrepancy-based awareness signal.
- `run_scale_sweep.sh` — sweeps `--lora_scale` ∈ {0.0, 0.2, …, 1.5} for a small test set, then runs `aggregate_results.py`.
- `collect_and_analyze_discrepancy.sh` — populates the discrepancy JSON for `analyze_discrepancy.py` to consume.

There is no test suite, linter, or CI configured.

## Architecture

### Train-time output layout (important)

`train.py` derives the run name from `Config.note` (and `lora.rank` when `lora.use_lora` is True), then writes to `trained_models/<note>[_rank_<rank>]/`. It also **copies `arguments.py` and `crowd_nav/configs/config.py` into that directory** so the run is reproducible.

`test.py` and `visualize.py` then **import `arguments.py` and `configs/config.py` from the model directory itself**, not the repo root, via `importlib.util.spec_from_file_location`. Changing the repo-root `arguments.py` does not affect testing an old checkpoint. To re-test with new args, edit the copies inside `trained_models/<NAME>/`.

### Two-critic PPO-Lagrangian

`train.py` builds two `Policy` instances: `actor_critic` (reward) and `cost_actor_critic` (cost). The PPO-Lagrangian loop lives in `rl/ppo/ppo_lag.py` and friends, with rollouts stored via `rl/networks/storage_safe.py`. The cost head enforces the `constrained_rl_related.cost_limit` constraint using a Lagrange multiplier (`rl/ppo/lagrange.py`).

### Environment stack

OpenAI-Gym envs subclass each other linearly: `CrowdSim → CrowdSimVarNum → CrowdSimPred → CrowdSimPredRealGST`. The env name in `arguments.py` (`--env-name`) selects which level you instantiate. With `sim.predict_method = 'inferred'`, vec envs are wrapped in `VecPretextNormalize` and the GST predictor at `gst_updated/results/...` is required; with any other prediction mode, wrapping is disabled. `config.py` enforces this consistency at import time.

`dt_aci/` provides the adaptive conformal inference used to compute per-step uncertainty intervals that feed the cost critic.

### Policy network (selfAttn_merge_srnn)

The default `robot.policy = 'selfAttn_merge_srnn'` resolves to `rl/networks/networkss.py` (HH self-attention → HR cross-attention → GRU temporal memory → actor/critic heads), wrapped by `rl/networks/model.py::Policy`. ORCA and Social Force baselines (`robot.policy in ['orca', 'social_force']`) bypass the neural net entirely — `actor_critic = None` and the policy decisions come from `crowd_nav/policy/`.

`rl/networks/ARCHITECTURE.md` has the canonical per-layer breakdown including which layers carry LoRA adapters and the old→new variable name map used to load legacy checkpoints.

### LoRA: training-time vs. inference-time

Set `lora.use_lora = True` in `crowd_nav/configs/config.py` to train with LoRA adapters wrapping select linear layers (defined in `rl/networks/network_utils.py::LoRALinear` / `LoRAAdapter`). Each adapter has a `dynamic_scale` attribute that multiplies the low-rank branch at forward time.

At test/visualize time, `--lora_behaviour` controls how `dynamic_scale` is set each step:
- `always_off` / `always_on` / `fixed_scale` — static (uses `--lora_scale`).
- `switching_{gt,discrepancy,discrepancynew,pred}` — discrete 0/1 switch driven by ground-truth awareness, discrepancy score against observed motion, or the `FriendlyPredictor` (`alpha_predictor.py`).
- `adaptive_{gt,discrepancy,discrepancynew,pred}` — continuous scale interpolated from the same signals.

The orchestration lives in `rl/evaluation.py`. The env's `baseEnv.robot.lora_scale` and the modules' `dynamic_scale` are both updated each step. `--adaptive_lora_scenario` selects a fixed PoC scenario where humans deterministically flip aware/ignorant (e.g. half-and-half, switch at step 25) — used by `test_adaptive_lora_poc.sh`.

When loading a non-LoRA checkpoint into a LoRA-enabled model, `test.py` remaps `*.weight` → `*.base_layer.weight` and loads with `strict=False`.

### Friendly/aware predictor

`alpha_predictor.py` defines `FriendlyPredictor` (per-human binary classifier "is this human aware of the robot?"). `train_alpha_predictor.py` trains it from JSON episode dumps (using each human's `actual_friendly` ground-truth field) and writes the weights to `<model_dir>/friendly_predictor.pth` alongside a `friendly_predictor_metrics.json` sidecar that records `human_dim`/`robot_dim`/val metrics. The `*_pred` LoRA behaviours load that checkpoint at test time via `rl/evaluation.py::_maybe_load_friendly_predictor`. (The unrelated repo-root `scenario_classifier.pt` is the output of the legacy `train_classifier_json.py` pipeline and is *not* used by the LoRA path.)

## Conventions worth knowing

- `Config.note` is the canonical run identifier — set it before training. Existing `trained_models/` names follow `LoraX_<visibility>[..._alpha_<n>]_rank_<r>` or `Fulltune_<...>` patterns.
- `*.png`, `*.csv`, `*.json`, `trained_models/`, `visualizations/`, `result/`, and `wandb/` are gitignored; analysis outputs from `aggregate_results.py`, `plot_*.py`, and `analyze_discrepancy.py` are not version-controlled.
- `--use-wandb` is opt-in; otherwise `WANDB_MODE=disabled` is set in `train.py`.
- `arguments.py` has an "frequently tuned" block at the top (seed, num-processes, num-mini-batch, clip-param, num-env-steps); the rest of the file is rarely touched.

## Analysis pipeline

After a sweep writes its CSV/JSON outputs, figures come from a small family of scripts at the repo root (none auto-run from training):
- `aggregate_results.py` — collapses per-run output files into a single summary table.
- `plot_experiment_results.py`, `plot_lora_results.py`, `plot_scale_results.py`, `plot_alpha_results.py`, `plot_progress.py`, `plot_training_time_comparison.py` — single-purpose plotters; pick the one matching your sweep.
- `analyze_discrepancy.py` — pairs with `collect_and_analyze_discrepancy.sh`.
- `results_visualization.ipynb` — interactive companion to the above; the canonical "load a finished sweep and draw the paper figures" notebook.

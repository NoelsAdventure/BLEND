# Repository Guidelines

## Project Structure & Module Organization

BLEND is a Python crowd-navigation research codebase. Core robot policies live in `crowd_nav/`, with environment implementations in `crowd_sim/`. Reinforcement-learning algorithms, rollout storage, vectorized environments, and network definitions are under `rl/`. Configuration entry points are `arguments.py`, `configs/config.py`, and `crowd_nav/configs/config.py`. Pretrained checkpoints and copied run configs are stored in `trained_models/`. `Python-RVO2/` contains the bundled ORCA/RVO2 extension, while `gst_updated/` contains the GST prediction model code. Top-level scripts such as `train.py`, `test.py`, `visualize.py`, and plotting/sweep scripts drive experiments. Generated logs, plots, notebooks, and temporary run folders should stay out of focused code changes unless the task explicitly concerns results.

## Build, Test, and Development Commands

Build the recommended container:

```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t blend:latest .
```

Run it with GPUs and the repo mounted:

```bash
docker run --gpus '"device=0"' --shm-size=32g -it -p 1234:8888 -v /home/docker_share:/home/dockeruser/shared -v $(pwd):/workspace blend:latest /bin/bash
```

Evaluate pretrained models with `python test.py`, create visualizations with `python visualize.py`, and train with `python train.py` after adjusting `arguments.py` and `crowd_nav/configs/config.py`. Scenario scripts such as `test_baselines.sh`, `run_all_results.sh`, and `run_scale_sweep.sh` wrap common experiment batches.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Follow the existing module pattern: configuration values are grouped in config classes, policy implementations belong in `crowd_nav/policy/`, and environment changes belong in `crowd_sim/envs/`. Keep experiment-specific constants near the script that consumes them. Avoid committing generated `__pycache__`, large transient logs, or ad hoc result artifacts.

## Testing Guidelines

There is no single project-wide pytest suite. Use `python test.py --model_dir <dir> --test_model <checkpoint>` for policy evaluation and targeted shell scripts for scenario coverage. The vendored `baselines/` package includes pytest tests; run focused tests with `python -m pytest baselines/common/tests` when modifying that code. Name new tests `test_*.py` and keep fixtures small enough to run without full training.

## Commit & Pull Request Guidelines

Recent history uses short, informal commit summaries and merge commits. Prefer concise imperative messages with enough scope to identify the experiment or module, for example `fix lora scale evaluation`. Pull requests should describe the behavioral change, list commands run, note config/checkpoint dependencies, and include plots or screenshots when results or visualization output change.

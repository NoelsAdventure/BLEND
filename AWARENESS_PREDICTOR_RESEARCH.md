# Awareness Predictor — Research Guide

Supervisor doc for any Claude (or human) working on improving the
`FriendlyPredictor` model. Read this end-to-end before touching code.

## 1. What the model does

`FriendlyPredictor` (`alpha_predictor.py`) is a **per-human binary
classifier** that answers, at every simulator step:

> Is human *i* aware of the robot right now?

- Label space: `{0 = ignorant, 1 = aware}`.
- Ground truth at simulation time: `baseEnv.robot.visible_to_humans[i]`
  (a fixed per-episode flag controlled by the scenario layout) — dumped
  into each JSON step as `humans[i].actual_friendly`.
- Consumer: `rl/evaluation.py::_compute_pred_friendly_probs` →
  `_compute_is_friendly` → the `*_pred` LoRA behaviours
  (`switching_pred`, `adaptive_pred`). Each step the predicted
  per-human probability drives `dynamic_scale` of the LoRA adapters,
  which switches the policy between "humans see me" and "humans
  ignore me" behaviour.

**Why we care:** the whole *_pred LoRA pipeline is downstream of this
model. A 5-point accuracy gain here typically converts to materially
fewer collisions in the invisible/aware-mix scenarios. The discrepancy
baseline (`*_discrepancy` behaviours) is the thing to beat.

## 2. Current baseline

Checkpoint: `trained_models/LoraF_invi_visi_rank_1/friendly_predictor.pth`
Sidecar:   `trained_models/LoraF_invi_visi_rank_1/friendly_predictor_metrics.json`

Last recorded metrics on the held-out episode split (epoch 40):

| Metric        | Value  |
|---------------|--------|
| Val accuracy  | 0.8392 |
| Val F1        | 0.8364 |
| Val precision | 0.8514 |
| Val recall    | 0.8219 |

Architecture currently saved: transformer, `hidden_dim=128`,
`num_heads=4`, `num_layers=2`, `dropout=0.1`.
(Note: `train_alpha_predictor.py` currently hard-codes `hidden_dim=192,
num_layers=3` — the checkpoint above pre-dates that change. Retrain to
match.)

Feature dims (must match at inference time):
`HUMAN_FEATURE_DIM = 24` (12 pred_traj + 5 uncertainty + 7 extras),
`ROBOT_FEATURE_DIM = 9`, `max_humans = 20`.

## 3. Contracts you must NOT break

These are the train/inference coupling points. Break any of these and
the predictor silently degrades at test time even if val metrics look
fine.

1. **Feature schema is shared.** `build_human_feature_row` and
   `build_robot_feature_vector` live in `train_alpha_predictor.py` and
   are imported by `rl/evaluation.py`. If you change the schema, both
   call sites change together, and `HUMAN_FEATURE_DIM` /
   `ROBOT_FEATURE_DIM` must update too.
2. **Distance-sorted human ordering.** The env's `talk2Env` publishes
   `out_pred` / `aci_predicted_conformity_scores` in distance-sorted
   order. The dataset sorts the same way. Do not change one without
   the other — feature row *i* must correspond to the same human at
   train and test time.
3. **Padding mask semantics.** `key_padding_mask=True` means *ignore
   this slot*. Loss masks padded slots out via `b_valid.float()`. Any
   new loss term must respect this or padded slots will pollute
   gradients toward "ignorant" (because we zero-fill padded labels).
4. **Sidecar is the source of truth for dims.** `_maybe_load_friendly_predictor`
   reads `human_dim`/`robot_dim` from `friendly_predictor_metrics.json`
   to build the inference-time module. If you bump the architecture,
   the sidecar `architecture` block must match what `FriendlyPredictor.__init__`
   was called with at training time, *and* `human_dim`/`robot_dim` must
   reflect actual feature widths.
5. **Label field name.** Training reads `actual_friendly` (falls back
   to legacy `is_friendly`). Don't rename without grepping every dump
   script.

## 4. Data pipeline

JSON dumps live under `trained_models/<run>/test/` and are produced by
running `test.py` / the scenario sweep with `--lora_behaviour
adaptive_gt`. The four scenarios currently used for training
(`scenarios_spec` in `train_alpha_predictor.py`):

- `seperate_mixed_5050` — half aware / half ignorant. The only
  scenario that forces real **per-human** discrimination. The most
  informative one — accuracy here is the headline number.
- `seperate_all_aware` — degenerate-positive.
- `seperate_all_ignorant` — degenerate-negative.
- `seperate_ignorant_to_aware_step25` — population flips at step 25.
  Tests *temporal* sensitivity (the current architecture is
  step-independent, so it can only react via observed motion).

Each scenario has a base + `_exp1` seed variant. Episode-level split
(20% val) guarantees no within-episode leakage.

Training uses `WeightedRandomSampler` to balance per-scenario
contribution per epoch. Model selection criterion is
**max-min-scenario-accuracy** (the worst scenario decides the
checkpoint), not mean — by design, to avoid degenerate solutions that
ace `all_aware` + `all_ignorant` and fail `mixed_5050`.

## 5. Levers to pull

In rough order of expected payoff. Start with (a) and (b) before
touching architecture.

### (a) Features
Per-human input is currently:
- 12-d `pred_traj` (6 future positions from GST, robot-frame offset,
  world-aligned).
- 5-d ACI uncertainty (per-step conformity scores).
- 7-d extras: `rel_x, rel_y, hvx, hvy, dist, approach_rate, h_radius`.

Things worth trying:
- **History.** Currently the model sees only the *current* step. Add
  a short observation window (e.g. last 5 steps of relative position
  / velocity, or trajectory residuals) so the network can read
  "human is/isn't deflecting away from the robot". This is the
  signal that `_discrepancy` exploits — the network should subsume it.
- **Heading alignment.** Cosine of (human velocity, robot→human
  vector) and (human velocity, predicted-human-velocity) — both are
  cheap and directly encode whether the human is reacting.
- **Egocentric rotation.** Rotate every per-human feature into the
  robot's heading frame so the model is invariant to global heading.
- **GST disagreement features.** The discrepancy score
  `||P_human − P_pred_before|| · |sin(ψ)|` is already computed in
  `evaluation.py:512`. Feeding it as an explicit input is a strong
  ablation lower bound.

### (b) Labels & data
- **Class balance per scenario.** Currently fine via the weighted
  sampler. But the per-human label is *temporally constant* within an
  episode under all current scenarios except `switch_step25`. Adding
  a true mid-episode-flip scenario (each human flips
  independently at a random time) would force the model to use
  temporal evidence rather than memorising layout priors.
- **More episodes.** Each scenario currently has ~hundreds of
  episodes (base + exp1). Cheap to generate more with
  `test.py --lora_behaviour adaptive_gt --test_size <N>` then
  re-train. Diminishing returns are not yet visible in the val curve.
- **Noisier "aware" humans.** If a human is aware but the robot is
  far away, they barely react — the label says "aware" but the
  feature row looks identical to an ignorant human. Consider a soft
  label that decays with distance, or just down-weight far humans in
  the loss.

### (c) Architecture
Cheap to try:
- **Bigger transformer.** Currently `hidden=192, layers=3, heads=4`
  in the script. Try `hidden=256, layers=4`. Watch for over-fit on
  `all_aware` / `all_ignorant`.
- **Temporal stack.** A 1-D conv or small GRU over the observation
  window (from lever (a)) before the cross-human transformer. This
  is the cleanest way to add memory.
- **Robot-as-token instead of broadcast-add.** Currently the robot
  embedding is added to every human token before attention. Try
  prepending it as a token-0 and reading it back the same way —
  often improves with deeper stacks.

### (d) Training
- **Focal loss or label smoothing.** Mixed-5050 has hard examples
  (humans far from the robot whose motion is ambiguous). Focal might
  help.
- **Longer schedule + early stop on min-scenario-acc.** Currently 60
  epochs cosine. Worth sweeping.
- **Eval threshold tuning.** Inference uses a fixed `0.5`. Sweeping
  on val and writing the chosen threshold into the sidecar (and
  reading it in `_compute_is_friendly`) is a free win if precision
  and recall are imbalanced.

### (e) The label itself
The current `actual_friendly` flag is *per-episode constant* (set by
`robot.visible_to_humans`). The interesting failure mode at deploy
time is humans who *transition* mid-episode. If you change the env to
emit a per-step label (e.g. "is the human currently in a deflection
cone around the robot, suggesting they noticed it"), retrain on that,
and gate it behind a config flag — keep the old behaviour available
for back-compat.

## 6. How to validate a change

Order of operations for any worker Claude:

1. **Sanity:** retrain with the *current* code on the *current* JSONs
   and confirm you can reproduce the ~0.84 baseline within ±0.01.
   If you can't, do not change anything else yet.
2. **Per-scenario table.** The script prints
   `==> Epoch N per-scenario acc: mixed_5050=… | all_aware=… | …`.
   The number that matters most is **mixed_5050**. The other three
   should stay above ~0.95 — if they drop, the model is over-fitting
   to whatever you added.
3. **End-to-end test.** After training, run the LoRA `_pred`
   behaviour on the test scenarios and compare collision / success /
   discomfort rates against the `_gt` and `_discrepancy` baselines.
   Use the existing sweep harness:
   ```bash
   ./test_adaptive_lora_poc.sh        # full PoC sweep
   # or, single scenario:
   python test.py --model_dir trained_models/<run>/ \
                  --test_model <ckpt>.pt \
                  --lora_behaviour adaptive_pred \
                  --adaptive_lora_scenario seperate_mixed_5050
   ```
   `_pred` should approach `_gt` and beat `_discrepancy`. If val
   accuracy went up but end-to-end metrics went down, the model is
   gaming val — investigate before claiming an improvement.
4. **Don't claim a win on val alone.** Aggregate the sweep with
   `aggregate_results.py` and report deltas vs. baseline.

## 7. Where outputs go

- Checkpoint: `<model_dir>/friendly_predictor.pth`
- Sidecar:    `<model_dir>/friendly_predictor_metrics.json`
- Training JSONs: `<model_dir>/test/seperate_*_adaptive_gt*.json`
- LoRA sweep results: `<model_dir>/test/` (CSV + JSON per behaviour /
  scenario)

`model_dir` is hard-coded to `trained_models/LoraF_invi_visi_rank_1`
in `train_alpha_predictor.py::train()`. If you fork a new predictor,
either parameterise this or change it explicitly — do not silently
overwrite the existing checkpoint until you've validated.

## 8. Common pitfalls — read before debugging

- **Feature-dim drift.** If `_compute_pred_friendly_probs` crashes
  with a dim mismatch at test time, the sidecar's `human_dim` /
  `robot_dim` disagrees with what `FriendlyPredictor` was built with.
  Rebuild and resave both.
- **Sort-order mismatch.** If acc looks fine on val but `_pred`
  behaves randomly at test time, you broke the distance-sort. Sort
  in *both* the dataset and `_compute_pred_friendly_probs` the same
  way.
- **Padding leakage.** If you add a global pooling layer (e.g. mean
  over humans for an episode-level head), it must respect the
  padding mask. Padded slots are not zero in the *feature space*
  after the encoder — only in the input.
- **Legacy dumps.** Some old JSONs miss `v_pref` and `actual_friendly`.
  The loader warns once and falls back. Don't train on a mix without
  noting which dumps are legacy — the fallback `is_friendly` is from
  an older labelling convention.
- **Don't import from the repo root at test time.** `test.py` and
  `visualize.py` import config from the model dir. The predictor
  path follows the same pattern via `model_dir`. Keep that.

## 9. What's out of scope here

- Changing the RL policy or LoRA adapter structure — that lives in
  `rl/networks/` and is the subject of `rl/networks/ARCHITECTURE.md`.
- Changing the GST trajectory predictor at `gst_updated/`.
- The `_discrepancy*` behaviours' scoring math. We benchmark against
  them; we don't tune them here.

If a proposed change requires editing any of the above, surface it
explicitly with a paragraph on why it's necessary — don't quietly
co-modify.

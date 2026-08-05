import json
import numpy as np
import torch
import os
import csv
from datetime import datetime
import uuid

from crowd_sim.envs.utils.info import *
from alpha_predictor import FriendlyPredictor
from train_alpha_predictor import (
    HUMAN_FEATURE_DIM, ROBOT_FEATURE_DIM, NUM_PRED_STEPS,
    build_human_feature_row,
)


def _maybe_load_friendly_predictor(behaviour, model_dir, device, logging, predictor_tag=None):
    # Resolve the predictor file path. With --predictor_tag T the loader
    # looks for friendly_predictor_T.pth + friendly_predictor_metrics_T.json
    # so different DAgger runs / training variants can coexist without
    # clobbering each other. Without a tag (default) it uses the canonical
    # friendly_predictor.pth.
    #
    # Tag semantics on MISSING file:
    #   - canonical (tag=None): error iff behaviour requires the predictor
    #     (anything with '_pred' in it), else return None (shadow eval off).
    #   - tagged: never error. Return the sentinel string 'SKIP_TAG_MISSING'
    #     so test.py can exit cleanly and let the shell sweep move on.
    if predictor_tag:
        predictor_path = os.path.join(model_dir, f'friendly_predictor_{predictor_tag}.pth')
        metrics_path = os.path.join(model_dir, f'friendly_predictor_metrics_{predictor_tag}.json')
    else:
        predictor_path = os.path.join(model_dir, 'friendly_predictor.pth')
        metrics_path = os.path.join(model_dir, 'friendly_predictor_metrics.json')

    if not os.path.exists(predictor_path):
        if predictor_tag:
            msg = (f"Tagged predictor not found: {predictor_path}. Skipping run "
                   f"so other tags in the sweep can proceed.")
            logging.warning(msg)
            print(msg)
            return 'SKIP_TAG_MISSING'
        if '_pred' in behaviour:
            raise FileNotFoundError(f"FriendlyPredictor weights not found at {predictor_path}")
        logging.info(f"No FriendlyPredictor weights at {predictor_path}; "
                     f"awareness accuracy will not be reported.")
        return None
    m = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as mf:
            m = json.load(mf)

    # Dims come from the metrics sidecar; fall back to the current feature
    # widths if the sidecar is missing.
    human_dim = int(m.get('human_dim', HUMAN_FEATURE_DIM))
    robot_dim = int(m.get('robot_dim', ROBOT_FEATURE_DIM))

    arch = m.get('architecture', {}) if isinstance(m, dict) else {}
    predictor = FriendlyPredictor(
        human_dim=human_dim, robot_dim=robot_dim,
        hidden_dim=int(arch.get('hidden_dim', 128)),
        num_heads=int(arch.get('num_heads', 4)),
        num_layers=int(arch.get('num_layers', 2)),
        dropout=float(arch.get('dropout', 0.1)),
    ).to(device)
    state = torch.load(predictor_path, map_location=device, weights_only=False)

    # Fail loudly on dim mismatch instead of silently mis-loading.
    # Probe the first linear of human_encoder for the saved human_dim.
    enc_w_key = 'human_encoder.0.weight'
    rbt_w_key = 'robot_encoder.0.weight'
    if enc_w_key in state and rbt_w_key in state:
        saved_human_dim = state[enc_w_key].shape[1]
        saved_robot_dim = state[rbt_w_key].shape[1]
        if (saved_human_dim, saved_robot_dim) != (human_dim, robot_dim):
            raise RuntimeError(
                f"FriendlyPredictor dim mismatch: checkpoint at {predictor_path} has "
                f"(human_dim={saved_human_dim}, robot_dim={saved_robot_dim}) but "
                f"metrics sidecar says ({human_dim}, {robot_dim}). Re-train or fix the sidecar."
            )

    predictor.load_state_dict(state)
    predictor.eval()
    logging.info(f"Loaded FriendlyPredictor from {predictor_path} "
                 f"(human_dim={human_dim}, robot_dim={robot_dim})")

    if m:
        msg = (f"FriendlyPredictor val metrics @ epoch {m.get('epoch', '?')}: "
               f"acc={m.get('val_accuracy', float('nan')):.4f} "
               f"f1={m.get('val_f1', float('nan')):.4f} "
               f"prec={m.get('val_precision', float('nan')):.4f} "
               f"rec={m.get('val_recall', float('nan')):.4f} "
               f"(TP={m.get('val_tp', '?')}, FP={m.get('val_fp', '?')}, "
               f"TN={m.get('val_tn', '?')}, FN={m.get('val_fn', '?')}, "
               f"N={m.get('val_size', '?')}, thr={m.get('threshold', '?')})")
        logging.info(msg)
        print(msg)
    else:
        logging.warning(f"No friendly_predictor_metrics.json found alongside {predictor_path}; "
                        f"retrain with the updated train_alpha_predictor.py to populate it.")
    return predictor


def _compute_pred_friendly_probs(friendly_predictor, out_pred, aci_predicted_conformity_scores,
                                 baseEnv, r_state_vec, device, max_humans=20, num_pred_steps=NUM_PRED_STEPS):
    # max_humans=20 mirrors train_alpha_predictor.py FriendlyDataset.
    # Feature layout mirrors build_human_feature_row in train_alpha_predictor.
    if friendly_predictor is None:
        return {}

    # aci_predicted_conformity_scores arrives as (num_humans, T) from the env
    # but is rewrapped to (1, num_humans, T) before this point.
    if aci_predicted_conformity_scores is None:
        uncs_2d = None
    else:
        uncs = np.asarray(aci_predicted_conformity_scores)
        while uncs.ndim > 2:
            uncs = uncs[0]
        uncs_2d = uncs  # (num_humans, T)

    # The env's talk2Env sorts humans by distance before publishing
    # out_pred / aci, so feature row i corresponds to the i-th
    # distance-sorted human. Mirror that order here.
    robot_pos = baseEnv.robot.get_position()
    sorted_humans = sorted(
        baseEnv.humans,
        key=lambda h: np.linalg.norm(np.array(h.get_position()) - np.array(robot_pos)),
    )

    # r_state_vec layout: [vx, vy, px, py, radius, gx, gy, v_pref, theta]
    rvx, rvy = float(r_state_vec[0]), float(r_state_vec[1])
    rx, ry = float(r_state_vec[2]), float(r_state_vec[3])

    h_states_list = []
    valid_mask = []
    for i in range(max_humans):
        if i < len(out_pred) and i < len(sorted_humans):
            traj = out_pred[i].tolist() if hasattr(out_pred[i], 'tolist') else list(out_pred[i])
            if uncs_2d is not None and i < uncs_2d.shape[0]:
                u_row = [float(x) for x in uncs_2d[i].tolist()]
            else:
                u_row = [0.0] * num_pred_steps
            h = sorted_humans[i]
            h_row = build_human_feature_row(
                traj, u_row,
                float(h.px), float(h.py),
                float(h.vx), float(h.vy),
                float(getattr(h, 'radius', 0.3)),
                rx, ry, rvx, rvy,
            )
            h_states_list.append(h_row)
            valid_mask.append(True)
        else:
            h_states_list.append([0.0] * HUMAN_FEATURE_DIM)
            valid_mask.append(False)

    h_states = torch.tensor([h_states_list], dtype=torch.float32).to(device)
    r_state = torch.tensor([[float(x) for x in r_state_vec]], dtype=torch.float32).to(device)
    key_padding_mask = torch.tensor([[not v for v in valid_mask]], dtype=torch.bool).to(device)

    with torch.no_grad():
        logits = friendly_predictor(h_states, r_state, key_padding_mask=key_padding_mask)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    result = {}
    for i, h in enumerate(sorted_humans):
        if i < max_humans:
            result[h.id] = float(probs[i])
    return result


def _compute_is_friendly(behaviour, human, human_idx, baseEnv,
                        discrepancy_scores, pred_friendly_probs,
                        human_above_threshold_count, test_args):
    if hasattr(baseEnv.robot, 'visible_to_humans'):
        actual_friendly = bool(baseEnv.robot.visible_to_humans[human_idx])
    else:
        actual_friendly = bool(baseEnv.robot.visible)

    if '_gt' in behaviour:
        is_friendly = actual_friendly
    elif '_pred' in behaviour:
        prob = pred_friendly_probs.get(human.id, 0.0)
        is_friendly = prob > 0.5
    else:
        score = discrepancy_scores.get(human.id, 0.0)
        threshold = getattr(test_args, 'discrepancy_threshold', 0.05)
        m_threshold = getattr(test_args, 'discrepancy_m', 1)
        if score > threshold:
            human_above_threshold_count[human.id] = human_above_threshold_count.get(human.id, 0) + 1
        else:
            human_above_threshold_count[human.id] = 0
        is_friendly = human_above_threshold_count.get(human.id, 0) >= m_threshold

    return is_friendly, actual_friendly


def _has_lora_modules(actor_critic):
    """True iff the model actually carries LoRA adapters. Replaces the older
    string-match-on-model_dir heuristic so behaviour doesn't depend on naming.
    """
    if actor_critic is None:
        return False
    from rl.networks.network_utils import LoRALinear, LoRAAdapter
    return any(isinstance(m, (LoRALinear, LoRAAdapter)) for m in actor_critic.modules())


def _apply_cluster_layout(baseEnv, cluster_spread=1.5, goal_jitter=1.0):
    """Override per-episode human positions/goals so that humans form two
    diametrically opposite spatial clusters that walk toward each other.
    Cluster 0 (first half by index) is ignorant; cluster 1 (second half) is
    aware. Called after envs.reset(); the first observation will still
    reflect pre-reposition state, but every subsequent step is correct.
    """
    n = len(baseEnv.humans)
    if n == 0:
        baseEnv.robot.visible_to_humans = []
        return
    half = n // 2

    circle_r = baseEnv.circle_radius
    theta_a = np.random.uniform(0, 2 * np.pi)
    center_a = np.array([circle_r * np.cos(theta_a), circle_r * np.sin(theta_a)])
    center_b = -center_a

    robot_pos = np.array(baseEnv.robot.get_position())
    robot_clearance = baseEnv.robot.radius + 0.5

    visible_flags = []
    for i, human in enumerate(baseEnv.humans):
        is_aware = i >= half
        spawn_center = center_b if is_aware else center_a
        goal_center = center_a if is_aware else center_b

        px, py = spawn_center
        for _ in range(50):
            offset = np.random.uniform(-cluster_spread, cluster_spread, 2)
            px, py = spawn_center + offset
            if np.linalg.norm([px - robot_pos[0], py - robot_pos[1]]) > human.radius + robot_clearance:
                break
        gx, gy = goal_center + np.random.uniform(-goal_jitter, goal_jitter, 2)

        human.px = float(px)
        human.py = float(py)
        human.gx = float(gx)
        human.gy = float(gy)
        human.vx = 0.0
        human.vy = 0.0
        visible_flags.append(is_aware)

    baseEnv.robot.visible_to_humans = visible_flags


def _apply_lora_scale(actor_critic, baseEnv, scale):
    """Single source of truth for keeping env-side state and every LoRA
    module's dynamic_scale in lock-step. Use this everywhere LoRA scale is
    set — avoids the lock-step drift class of bug.
    """
    scale = float(scale)
    baseEnv.robot.lora_scale = scale
    baseEnv.robot.lora_enabled = (scale > 0)
    if actor_critic is None:
        return
    from rl.networks.network_utils import LoRALinear, LoRAAdapter
    for module in actor_critic.modules():
        if isinstance(module, (LoRALinear, LoRAAdapter)):
            module.dynamic_scale = scale


def evaluate(actor_critic, eval_envs, num_processes, device, test_size, logging, config, args, model_dir, visualize=False, test_args=None, video_save_path=None):
    """ function to run all testing episodes and log the testing metrics """
    # initializations
    eval_episode_rewards = []
    
    # Awareness prediction accuracy tracking
    total_awareness_predictions = 0
    correct_awareness_predictions = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    all_discrepancy_data = {'aware': [], 'ignorant': []}

    behaviour = getattr(test_args, 'lora_behaviour', 'none')
    predictor_tag = getattr(test_args, 'predictor_tag', None) or None
    # Parse --render_only_cases into a set of ints; None = render all episodes.
    _roc = getattr(test_args, 'render_only_cases', None)
    if _roc:
        render_only_cases = {int(x) for x in _roc.split(',') if x.strip()}
    else:
        render_only_cases = None
    friendly_predictor = _maybe_load_friendly_predictor(behaviour, model_dir, device, logging,
                                                       predictor_tag=predictor_tag)
    if friendly_predictor == 'SKIP_TAG_MISSING':
        # Tagged predictor doesn't exist — gracefully exit so the calling
        # shell sweep can move on to the next combo without failing.
        msg = (f"Skipping evaluation: predictor_tag='{predictor_tag}' has no "
               f"matching file in {model_dir}.")
        logging.warning(msg)
        print(msg)
        return
    # When no tag is set, also accept the legacy None (predictor absent for
    # non-_pred behaviours) — fall through to the existing shadow-eval gating.

    # Awareness shadow-eval gating. 'always' (default) scores the predictor
    # against ground truth every step regardless of behaviour. 'pred_only'
    # restricts scoring to *_pred behaviours where the predictor is already
    # driving LoRA, so adaptive_gt / always_* runs don't pay the forward pass
    # or report stats. 'off' disables shadow eval entirely.
    awareness_eval = getattr(test_args, 'awareness_eval', 'always')
    if awareness_eval == 'off':
        eval_predictor_this_run = False
    elif awareness_eval == 'pred_only':
        eval_predictor_this_run = '_pred' in behaviour
    else:  # 'always'
        eval_predictor_this_run = True

    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           actor_critic.base.human_human_edge_rnn_size,
                                                                           device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []

    success = 0
    collision = 0
    timeout = 0
    too_close_ratios = []
    min_dist = []

    collision_cases = []
    timeout_cases = []

    all_path_len = []
    all_avg_uncertainty = []

    # Store detailed per-episode data
    episodes_data = []

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    # Experiment ID logic (moved up for video saving)
    user_exp_id = getattr(test_args, 'exp_id', None)
    scenario = getattr(test_args, 'adaptive_lora_scenario', 'none')
    behaviour = getattr(test_args, 'lora_behaviour', 'none')
    
    if scenario != 'none' or behaviour != 'none':
        base_id = f"{scenario}_{behaviour}"
        if user_exp_id:
            exp_id = f"{base_id}_exp{user_exp_id}"
        else:
            exp_id = base_id
    else:
        if user_exp_id:
            exp_id = f"exp{user_exp_id}"
        else:
            exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    # start the testing episodes
    try:
        from tqdm import tqdm
        pbar = tqdm(range(test_size), desc='Evaluating')
    except ImportError:
        pbar = range(test_size)

    for k in pbar:
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        obs = eval_envs.reset()
        out_pred = obs['spatial_edges'][:, :, :].to('cpu').numpy()[0]
        outs = baseEnv.talk2Env(out_pred)
        aci_predicted_conformity_scores, aci_cost = outs#np.array([o[0] for o in outs]) # [num_envs, num_humans, num_pred_steps]
        
        # Track uncertainty for the episode
        episode_uncertainties = []
        if aci_predicted_conformity_scores is not None and len(aci_predicted_conformity_scores) > 0:
            episode_uncertainties.append(np.mean(aci_predicted_conformity_scores))

        aci_predicted_conformity_scores = np.array([aci_predicted_conformity_scores])
        obs['conformity_scores'] = torch.from_numpy(aci_predicted_conformity_scores).to(torch.float32).to(device)
        global_time = 0.0
        path_len = 0.
        too_close = 0.
        last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

        if config.robot.policy not in ['orca', 'social_force']:
            eval_recurrent_hidden_states = {}

            node_num = 1
            edge_num = actor_critic.base.human_num + 1
            eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                        device=device)

            eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                            actor_critic.base.human_human_edge_rnn_size,
                                                                            device=device)
        
        # Adaptive LoRA Proof of Concept: Initialization
        scenario = getattr(test_args, 'adaptive_lora_scenario', 'none')
        behaviour = getattr(test_args, 'lora_behaviour', 'none')
        
        # Default lora_scale from args
        baseEnv.robot.lora_scale = getattr(test_args, 'lora_scale', 1.0)

        # Plumb behaviour + LoRA-capability onto the env so plot_scenario can
        # (a) show the dynamic scale only for adaptive_* runs and (b) recolor
        # non-LoRA baselines (e.g. FullFineTune) red instead of the
        # lora_scale=0 yellow default.
        baseEnv.lora_behaviour = behaviour
        baseEnv.is_lora_model = bool(getattr(getattr(config, 'lora', None), 'use_lora', False))

        # Clean up any leftover PoC attributes from previous episodes
        if hasattr(baseEnv.robot, 'visible_to_humans'):
            delattr(baseEnv.robot, 'visible_to_humans')

        # Behaviour Overrides
        if behaviour == 'always_on':
            _apply_lora_scale(actor_critic, baseEnv, 1.0)
        elif behaviour == 'always_off':
            _apply_lora_scale(actor_critic, baseEnv, 0.0)
        elif behaviour == 'fixed_scale':
            _apply_lora_scale(actor_critic, baseEnv, getattr(test_args, 'lora_scale', 1.0))


        # Behaviours that drive the per-step adaptive/switching loop.
        _ADAPTIVE_BEHAVIOURS = {
            'switching_gt', 'switching_discrepancy', 'switching_discrepancynew', 'switching_pred',
            'adaptive_gt', 'adaptive_discrepancy', 'adaptive_discrepancynew', 'adaptive_pred',
        }

        if scenario == 'seperate_mixed_5050':
            # One group (half) is ignorant (False), the other is friendly (True)
            baseEnv.robot.visible_to_humans = [False if i < len(baseEnv.humans)//2 else True
                                               for i in range(len(baseEnv.humans))]
            if k == 0:
                msg = f"Group scenario: {sum(not v for v in baseEnv.robot.visible_to_humans)} ignorant, {sum(baseEnv.robot.visible_to_humans)} friendly"
                if hasattr(pbar, 'write'):
                    pbar.write(msg)
                else:
                    print(msg)

            # Mirror the other scenario branches: explicitly init LoRA state
            # for switching/adaptive behaviours so the modules don't carry a
            # stale dynamic_scale from the previous episode. Aggressive (1.0)
            # default — the per-step loop will refine after step 0 anyway.
            if behaviour in _ADAPTIVE_BEHAVIOURS:
                _apply_lora_scale(actor_critic, baseEnv, 1.0)
            else:
                # Non-adaptive behaviour: render flag reflects whether the
                # model actually carries LoRA modules, not a path-string match.
                baseEnv.robot.lora_enabled = _has_lora_modules(actor_critic)

        elif scenario == 'seperate_ignorant_to_aware_step25':
            # Start invisible (Ignorant humans)
            baseEnv.robot.visible = False
            # Start with LoRA OFF for switching/adaptive — will flip ON at step 25.
            if behaviour in _ADAPTIVE_BEHAVIOURS:
                _apply_lora_scale(actor_critic, baseEnv, 0.0)

        elif scenario == 'seperate_all_ignorant':
            baseEnv.robot.visible = False
            if behaviour in _ADAPTIVE_BEHAVIOURS:
                _apply_lora_scale(actor_critic, baseEnv, 0.0)

        elif scenario == 'seperate_all_aware':
            baseEnv.robot.visible = True
            if behaviour in _ADAPTIVE_BEHAVIOURS:
                _apply_lora_scale(actor_critic, baseEnv, 1.0)

        elif scenario == 'cluster_aware_ignorant':
            _apply_cluster_layout(baseEnv)
            if k == 0:
                msg = (f"Cluster scenario: {sum(not v for v in baseEnv.robot.visible_to_humans)} ignorant "
                       f"(cluster A), {sum(baseEnv.robot.visible_to_humans)} aware (cluster B)")
                if hasattr(pbar, 'write'):
                    pbar.write(msg)
                else:
                    print(msg)
            if behaviour in _ADAPTIVE_BEHAVIOURS:
                _apply_lora_scale(actor_critic, baseEnv, 1.0)
            else:
                baseEnv.robot.lora_enabled = _has_lora_modules(actor_critic)

        # Initialize prev_predictions before the loop for tracking discrepancy
        prev_predictions = {}
        human_above_threshold_count = {} # Track consecutive frames above threshold
        for human in baseEnv.humans:
            if human.last_prediction is not None and len(human.last_prediction) > 1:
                prev_predictions[human.id] = {
                    'past_human': human.last_prediction[0],
                    'past_pred': human.last_prediction[1]
                }

        episode_steps = []
        episode_uncertainties = []
        episode_lora_scales = []
        episode_discrepancies = []
        episode_friendly_flags = []
        while not done:
            stepCounter = stepCounter + 1
            
            # Robot features in robot_node: [px, py, radius, gx, gy, v_pref, theta]
            # Robot features in temporal_edges: [vx, vy]
            r_node = obs['robot_node'][0, 0].cpu().numpy()
            r_vel = obs['temporal_edges'][0, 0].cpu().numpy()
            
            # Calculate Prediction Discrepancy Scores to guess who is aware/ignorant
            discrepancy_scores = {}
            robot_pos = baseEnv.robot.get_position()
            
            for i, human in enumerate(baseEnv.humans):
                if human.id in prev_predictions:
                    actual_human_pos = np.array([human.px, human.py])
                    
                    if isinstance(prev_predictions[human.id], dict):
                        predicted_human_pos = prev_predictions[human.id]['past_pred']
                        past_human_pos = prev_predictions[human.id]['past_human']
                    else:
                        predicted_human_pos = prev_predictions[human.id]
                        past_human_pos = None
                    
                    # Calculate vectors
                    v_robot_pred = np.array(robot_pos) - predicted_human_pos
                    v_human_pred = actual_human_pos - predicted_human_pos

                    # The two `_discrepancy*` variants compute psi differently
                    # but share downstream thresholding in _compute_is_friendly:
                    #   _discrepancy    — angle between (robot→pred) and (human→pred).
                    #                     Asks "did the human deviate away from
                    #                     the robot relative to the prediction?"
                    #   _discrepancynew — angle between past-human-motion and
                    #                     (pred→actual). Asks "did the human's
                    #                     turn rate / direction shift between
                    #                     the past step and now?"
                    if 'discrepancynew' in behaviour and past_human_pos is not None:
                        v_past_human_to_pred = predicted_human_pos - past_human_pos
                        v_pred_to_actual = actual_human_pos - predicted_human_pos

                        angle_past = np.arctan2(v_past_human_to_pred[1], v_past_human_to_pred[0])
                        angle_actual = np.arctan2(v_pred_to_actual[1], v_pred_to_actual[0])
                        psi = angle_actual - angle_past
                    else:
                        angle_robot = np.arctan2(v_robot_pred[1], v_robot_pred[0])
                        angle_human = np.arctan2(v_human_pred[1], v_human_pred[0])
                        psi = angle_human - angle_robot

                    # Calculate discrepancy score: ||P_human - P_pred_before|| * abs(sin(psi))
                    discrepancy_score = np.linalg.norm(v_human_pred) * abs(np.sin(psi))                        
                    discrepancy_scores[human.id] = discrepancy_score

                    # Track ground truth for statistics collection
                    if hasattr(baseEnv.robot, 'visible_to_humans'):
                        actual_friendly = baseEnv.robot.visible_to_humans[i]
                        if actual_friendly:
                            all_discrepancy_data['aware'].append(float(discrepancy_score))
                        else:
                            all_discrepancy_data['ignorant'].append(float(discrepancy_score))
                    else:
                        # If visibility not set per human, check overall robot.visible
                        if baseEnv.robot.visible:
                            all_discrepancy_data['aware'].append(float(discrepancy_score))
                        else:
                            all_discrepancy_data['ignorant'].append(float(discrepancy_score))

            # Adaptive LoRA Proof of Concept: Mid-episode switch for 'seperate_ignorant_to_aware_step25' scenario
            if scenario == 'seperate_ignorant_to_aware_step25' and stepCounter == 25:
                baseEnv.robot.visible = True
                if behaviour in _ADAPTIVE_BEHAVIOURS:
                    if test_size <= 10:
                        print(f"\n>>> Step {stepCounter}: Switching to VISIBLE and LoRA ON (Adaptive PoC)")
                    _apply_lora_scale(actor_critic, baseEnv, 1.0)

            # Cache of per-step is_friendly decisions, reused by the JSON logger below
            step_is_friendly_by_id = {}

            # Match the RL policy's robot input: concat([temporal_edges,
            # robot_node]) = [vx, vy, px, py, radius, gx, gy, v_pref, theta]
            # (see rl/networks/networkss.py:197). Hoisted out of the
            # behaviour-specific branch below so the shadow predictor eval
            # can run regardless of which behaviour drives the policy.
            r_state_vec = [
                float(r_vel[0]), float(r_vel[1]),
                float(r_node[0]), float(r_node[1]), float(r_node[2]),
                float(r_node[3]), float(r_node[4]), float(r_node[5]),
                float(r_node[6]),
            ]
            # Shadow predictor eval: when the predictor weights live in the
            # model dir, run them every step (whatever the behaviour) and
            # score against ground-truth visibility for every in-range
            # human. This populates the AwAcc / AwF1 / TP-FP-TN-FN stats
            # shown in the progress bar, so `adaptive_gt` etc. report the
            # predictor quality too without affecting the policy.
            #
            # Skip the forward pass entirely when neither the policy needs it
            # (`*_pred` behaviours) nor shadow-eval is enabled. This avoids
            # wasted compute AND keeps non-pred runs (e.g. fixed_scale
            # ablation) tolerant of feature-dim mismatches between the
            # build_human_feature_row() in train_alpha_predictor.py and the
            # on-disk predictor checkpoint.
            _predictor_needed = (
                friendly_predictor is not None and
                ('_pred' in behaviour or eval_predictor_this_run)
            )
            if _predictor_needed:
                pred_friendly_probs = _compute_pred_friendly_probs(
                    friendly_predictor, out_pred, aci_predicted_conformity_scores,
                    baseEnv, r_state_vec, device,
                )
            else:
                pred_friendly_probs = {}
            if friendly_predictor is not None and eval_predictor_this_run:
                _robot_pos_shadow = baseEnv.robot.get_position()
                for _i, _human in enumerate(baseEnv.humans):
                    _dist = np.linalg.norm(
                        np.array(_human.get_position()) - np.array(_robot_pos_shadow)
                    )
                    if _dist > baseEnv.robot.sensor_range:
                        continue
                    # The predictor only scored the top-max_humans closest
                    # (see _compute_pred_friendly_probs). Humans beyond that
                    # cap have no real prediction; counting them as "0.0 ⇒
                    # predicted ignorant" silently inflates FN/TN. Skip them
                    # so AwAcc reflects only what the predictor actually said.
                    if _human.id not in pred_friendly_probs:
                        continue
                    if hasattr(baseEnv.robot, 'visible_to_humans'):
                        _actual = bool(baseEnv.robot.visible_to_humans[_i])
                    else:
                        _actual = bool(baseEnv.robot.visible)
                    _prob = pred_friendly_probs[_human.id]
                    _pred = _prob > 0.5
                    total_awareness_predictions += 1
                    if _pred == _actual:
                        correct_awareness_predictions += 1
                    if _pred and _actual:
                        tp += 1
                    elif _pred and not _actual:
                        fp += 1
                    elif not _pred and _actual:
                        fn += 1
                    else:
                        tn += 1

            # Continuous adaptive scale or majority-based switching
            if behaviour in ['switching_gt', 'switching_discrepancy', 'switching_discrepancynew', 'switching_pred', 'adaptive_gt', 'adaptive_discrepancy', 'adaptive_discrepancynew', 'adaptive_pred']:
                robot_pos = baseEnv.robot.get_position()
                robot_theta = obs['robot_node'][0, 0, 6].item() # robot heading

                total_weight = 0.0
                friendly_weight = 0.0
                friendly_in_range = 0
                humans_in_range_count = 0

                for i, human in enumerate(baseEnv.humans):
                    dist = np.linalg.norm(np.array(human.get_position()) - np.array(robot_pos))
                    if dist <= baseEnv.robot.sensor_range:
                        humans_in_range_count += 1

                        is_friendly, actual_friendly = _compute_is_friendly(
                            behaviour, human, i, baseEnv,
                            discrepancy_scores, pred_friendly_probs,
                            human_above_threshold_count, test_args,
                        )
                        step_is_friendly_by_id[human.id] = bool(is_friendly)

                        if is_friendly:
                            friendly_in_range += 1
                            
                        # Calculate distance-based weight (1.0 at dist=0, 0.0 at dist=sensor_range)
                        dist_weight = max(0.0, 1.0 - (dist / baseEnv.robot.sensor_range))
                        
                        # Calculate directional weight (cos of half angle difference)
                        angle_to_human = np.arctan2(human.py - robot_pos[1], human.px - robot_pos[0])
                        theta_i = robot_theta - angle_to_human
                        
                        # Ensure theta_i is in [-pi, pi] for consistent cos(theta/2) behavior
                        theta_i = (theta_i + np.pi) % (2 * np.pi) - np.pi
                        direction_weight = np.cos(theta_i / 2)
                        
                        weight = dist_weight * direction_weight
                        
                        total_weight += weight
                        if is_friendly:
                            friendly_weight += weight
                
                # Calculate target lora_scale.
                # No humans in range, or (for adaptive) every in-range human is
                # ~directly behind the robot (total_weight ≈ 0), means there's
                # no crowd actually constraining the robot's path. Default to
                # aggressive (scale = 1.0) rather than snapping LoRA off — the
                # robot doesn't know yet if humans will appear ahead, and being
                # ready to handle oblivious ones is the safer prior.
                if humans_in_range_count == 0:
                    target_scale = 1.0
                    ratio = 1.0
                elif 'switching' in behaviour:
                    ratio = friendly_in_range / humans_in_range_count
                    target_scale = 1.0 if ratio > 0.5 else 0.0
                else:  # adaptive
                    if total_weight > 0:
                        ratio = friendly_weight / total_weight
                        target_scale = ratio
                    else:
                        # All in-range humans are behind the robot — direction
                        # weights collapsed to ~0. Treat the same as empty.
                        target_scale = 1.0
                        ratio = 1.0
                
                # Apply change if scale differs significantly (deadband ~0.01
                # prevents per-step jitter on near-stable targets).
                if abs(target_scale - baseEnv.robot.lora_scale) > 0.01:
                    if test_size <= 10:
                        msg = f"Ratio {ratio:.2f}"
                        if 'switching' in behaviour:
                            msg = f"Ratio {friendly_in_range}/{humans_in_range_count}={ratio:.2f}"
                        print(f"\n>>> Step {stepCounter}: {msg}. Target Scale = {target_scale:.2f}")
                    _apply_lora_scale(actor_critic, baseEnv, target_scale)
            
            # Collect data for the CURRENT step before taking the next action
            
            # Current human states
            human_states = []
            for i, h in enumerate(baseEnv.humans):
                # Always check for discrepancy if available
                h_discrepancy_raw = discrepancy_scores.get(h.id)

                # Reuse the decision the scaling loop just made for this step
                # (only populated for in-range humans; defaults to False otherwise).
                is_friendly = step_is_friendly_by_id.get(h.id, False)

                # Ground-truth awareness from the env, for ALL humans regardless
                # of sensor range. This is what train_alpha_predictor.py uses as
                # the label, and the prior fallback-to-False masked far-away
                # humans as ignorant during _gt dumps.
                if hasattr(baseEnv.robot, 'visible_to_humans'):
                    actual_friendly = bool(baseEnv.robot.visible_to_humans[i])
                else:
                    actual_friendly = bool(baseEnv.robot.visible)

                human_states.append({
                    'id': int(h.id),
                    'pos': [float(h.px), float(h.py)],
                    'vel': [float(h.vx), float(h.vy)],
                    'radius': float(h.radius),
                    'discrepancy': float(h_discrepancy_raw) if h_discrepancy_raw is not None else None,
                    'is_friendly': bool(is_friendly),
                    'actual_friendly': actual_friendly,
                })
                
                # Only include in averages if the data was actually available
                if h_discrepancy_raw is not None:
                    episode_discrepancies.append(float(h_discrepancy_raw))
                
                episode_friendly_flags.append(bool(is_friendly))

            step_data = {
                'step': stepCounter,
                'robot': {
                    'pos': [float(r_node[0]), float(r_node[1])],
                    'vel': [float(r_vel[0]), float(r_vel[1])],
                    'goal': [float(r_node[3]), float(r_node[4])],
                    'theta': float(r_node[6]),
                    'radius': float(r_node[2]),
                    'v_pref': float(r_node[5]),
                    'lora_scale': float(baseEnv.robot.lora_scale)
                },
                'humans': human_states,
                'pred_traj': out_pred.tolist(),
                'uncertainty': aci_predicted_conformity_scores[0].tolist() if aci_predicted_conformity_scores is not None else None
            }
            episode_steps.append(step_data)
            episode_lora_scales.append(float(baseEnv.robot.lora_scale))

            if config.robot.policy not in ['orca', 'social_force']:
                # run inference on the NN policy
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            else:
                action = torch.zeros([1, 2], device=device)
            if not done:
                global_time = baseEnv.global_time

            # if the vec_pretext_normalize.py wrapper is used, send the predicted traj to env
            if visualize:
                eval_envs.render()
                if video_save_path and (render_only_cases is None or k in render_only_cases):
                    baseEnv.plot_step(video_save_path)

            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)
            
            out_pred = obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
            # send manager action to all processes
            out_pred = obs['spatial_edges'][:, :, :].to('cpu').numpy()[0]
            outs = baseEnv.talk2Env(out_pred)
            aci_predicted_conformity_scores, aci_cost = outs#np.array([o[0] for o in outs]) # [num_envs, num_humans, num_pred_steps]
            
            # Update prev_predictions for the next step
            for h in baseEnv.humans:
                if h.last_prediction is not None and len(h.last_prediction) > 1:
                    prev_predictions[h.id] = {
                        'past_human': h.last_prediction[0],
                        'past_pred': h.last_prediction[1]
                    }
            
            # Track uncertainty for each step
            if aci_predicted_conformity_scores is not None and len(aci_predicted_conformity_scores) > 0:
                episode_uncertainties.append(np.mean(aci_predicted_conformity_scores))

            aci_predicted_conformity_scores = np.array([aci_predicted_conformity_scores])
            obs['conformity_scores'] = torch.from_numpy(aci_predicted_conformity_scores).to(torch.float32).to(device)
            # render

            # record the info for calculating testing metrics
            rewards.append(rew)

            path_len = path_len + np.linalg.norm(obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)
            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()


            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            episode_rew += rew[0]


            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        # an episode ends!
        if visualize:
            print('')
            print('Reward={}'.format(episode_rew))
            print('Episode', k, 'ends in', stepCounter)
        
        all_path_len.append(path_len)
        too_close_ratios.append(too_close/stepCounter*100)
        
        avg_uncertainty = np.mean(episode_uncertainties) if len(episode_uncertainties) > 0 else 0.0
        all_avg_uncertainty.append(avg_uncertainty)

        avg_lora_scale = np.mean(episode_lora_scales) if len(episode_lora_scales) > 0 else 0.0
        avg_discrepancy_ep = np.mean(episode_discrepancies) if len(episode_discrepancies) > 0 else 0.0
        avg_friendly_ratio_ep = np.mean(episode_friendly_flags) if len(episode_friendly_flags) > 0 else 0.0

        episode_result = 'Unknown'
        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            episode_result = 'Success'
            if visualize: print('Success')
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            episode_result = 'Collision'
            if visualize: print('Collision')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(time_limit)
            episode_result = 'Timeout'
            if visualize: print('Time out')
        
        episodes_data.append({
            'episode': k,
            'result': episode_result,
            'reward': float(episode_rew),
            'steps': stepCounter,
            'time': float(global_time),
            'path_length': float(path_len),
            'avg_uncertainty': float(avg_uncertainty),
            'avg_lora_scale': float(avg_lora_scale),
            'avg_discrepancy': float(avg_discrepancy_ep),
            'avg_friendly_ratio': float(avg_friendly_ratio_ep),
            'steps_data': episode_steps
        })

        if hasattr(pbar, 'set_postfix'):
            avg_lora_so_far = np.mean([ep['avg_lora_scale'] for ep in episodes_data])
            avg_pl_so_far = np.mean(all_path_len)
            postfix = {
                'SR': f'{success/(k+1):.2f}',
                'CR': f'{collision/(k+1):.2f}',
                'Avg PL': f'{avg_pl_so_far:.2f}',
                'Avg LoRA': f'{avg_lora_so_far:.2f}',
            }
            # Running awareness accuracy / F1, only populated when behaviour
            # generates per-human predictions (*_pred or *_discrepancy*).
            total_preds = tp + fp + tn + fn
            if total_preds > 0:
                acc_so_far = (tp + tn) / total_preds
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                postfix['AwAcc'] = f'{acc_so_far:.2f}'
                postfix['AwF1'] = f'{f1:.2f}'
            pbar.set_postfix(postfix)

        if not visualize and (k + 1) % 50 == 0:
            avg_sr = success / (k + 1)
            avg_cr = collision / (k + 1)
            avg_lora = np.mean([ep['avg_lora_scale'] for ep in episodes_data])
            summary_str = f"[Step {k+1}] SR: {avg_sr:.3f}, CR: {avg_cr:.3f}, Avg LoRA: {avg_lora:.3f}"
            total_preds = tp + fp + tn + fn
            if total_preds > 0:
                aw_acc = (tp + tn) / total_preds
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                aw_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                summary_str += (
                    f", AwAcc: {aw_acc:.3f}, AwF1: {aw_f1:.3f}, "
                    f"TP: {100*tp/total_preds:.1f}%, "
                    f"FP: {100*fp/total_preds:.1f}%, "
                    f"TN: {100*tn/total_preds:.1f}%, "
                    f"FN: {100*fn/total_preds:.1f}% (N={total_preds})"
                )
            if hasattr(pbar, 'write'):
                pbar.write(summary_str)
            else:
                print(f"\n{summary_str}")

        if video_save_path and (render_only_cases is None or k in render_only_cases):
            baseEnv.animate_episode(video_save_path, f"{exp_id}_ep{k}_{episode_result}",
                                    outcome=episode_result, avg_lora_scale=avg_lora_scale,
                                    behaviour=behaviour)

    if not visualize:
        print() # Move to next line after progress bar

    # all episodes end
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else time_limit  # baseEnv.env.time_limit

    # logging
    logging.info(
        'Testing success rate: {:.4f}, collision rate: {:.4f}, timeout rate: {:.4f}, '
        'nav time: {:.4f}, path length: {:.4f}, average intrusion ratio: {:.4f}%, '
        'average minimal distance during intrusions: {:.4f}, average prediction uncertainty: {:.4f}, '
        'average LoRA scale: {:.4f}'.
            format(success_rate, collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
                   np.mean(too_close_ratios), np.mean(min_dist), np.mean(all_avg_uncertainty),
                   np.mean([ep['avg_lora_scale'] for ep in episodes_data])))

    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    
    # JSON logic: Save a summary to the main file and full data to a unique experiment file
    lora_scale = getattr(test_args, 'lora_scale', 1.0)
    
    # Detailed config for the experiment
    important_config = {
        'robot_visible': config.robot.visible,
        'robot_fov': float(config.robot.FOV),
        'human_num': int(config.sim.human_num),
        'human_random': config.env.randomize_attributes,
        'human_fov': float(config.humans.FOV),
        'env_name': args.env_name,
        'use_lora': getattr(getattr(config, 'lora', object()), 'use_lora', False),
        'lora_alpha': getattr(getattr(config, 'lora', object()), 'alpha', None),
        'lora_rank': getattr(getattr(config, 'lora', object()), 'rank', None),
        'lora_scale': lora_scale,
        'test_model': getattr(test_args, 'test_model', None),
        'model_dir': model_dir
    }

    # Calculate average discrepancy scores for the experiment
    aware_scores = all_discrepancy_data.get('aware', [])
    ignorant_scores = all_discrepancy_data.get('ignorant', [])
    all_scores = aware_scores + ignorant_scores
    
    avg_discrepancy_aware = np.mean(aware_scores) if aware_scores else 0.0
    avg_discrepancy_ignorant = np.mean(ignorant_scores) if ignorant_scores else 0.0
    avg_discrepancy_all = np.mean(all_scores) if all_scores else 0.0

    full_experiment_data = {
        'exp_id': exp_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': important_config,
        'summary': {
            'num_episodes': test_size,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'avg_nav_time': avg_nav_time,
            'avg_path_length': float(np.mean(all_path_len)),
            # std fields needed for downstream ±CI computation (Wilson for SR,
            # mean±SE for PL/NavTime). Cheap one-pass numpy calls on
            # already-collected per-episode lists.
            'std_path_length': float(np.std(all_path_len, ddof=1)) if len(all_path_len) > 1 else 0.0,
            'std_nav_time': float(np.std(success_times, ddof=1)) if len(success_times) > 1 else 0.0,
            'std_uncertainty': float(np.std(all_avg_uncertainty, ddof=1)) if len(all_avg_uncertainty) > 1 else 0.0,
            # Intrusion rate (ITR, % of steps where robot was inside the
            # discomfort distance) — one value per episode in too_close_ratios.
            'avg_intrusion_ratio_pct': float(np.mean(too_close_ratios)) if too_close_ratios else 0.0,
            'std_intrusion_ratio_pct': float(np.std(too_close_ratios, ddof=1)) if len(too_close_ratios) > 1 else 0.0,
            # Social distance (SD, min robot-human distance during intrusions) —
            # one value per intrusion step in min_dist; NaN if no intrusions.
            'avg_min_social_distance': float(np.mean(min_dist)) if min_dist else float('nan'),
            'std_min_social_distance': float(np.std(min_dist, ddof=1)) if len(min_dist) > 1 else 0.0,
            'n_intrusion_episodes': int(len(min_dist)),
            'avg_uncertainty': float(np.mean(all_avg_uncertainty)),
            'avg_lora_scale': float(np.mean([ep['avg_lora_scale'] for ep in episodes_data])),
            'avg_discrepancy_aware': float(avg_discrepancy_aware),
            'avg_discrepancy_ignorant': float(avg_discrepancy_ignorant),
            'avg_discrepancy_all': float(avg_discrepancy_all),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        },

        'episodes': episodes_data
    }
    
    # 1. Save FULL per-episode dump.
    #    Default whitelist = behaviours whose JSONs are consumed by the
    #    training pipeline (train_alpha_from_json / dagger_loop):
    #       adaptive_gt    — supervised anchor data
    #       adaptive_pred  — DAgger closed-loop rollouts
    #       switching_pred — DAgger closed-loop rollouts (binary variant)
    #    Every other behaviour (always_off / always_on / *_discrepancy / etc.)
    #    skips this 100s-of-MB write — the summary block still goes into
    #    all_evaluations.json + the CSV below.
    #
    #    Override via test.py --save_episode_dump {auto|always|never}:
    #       auto    — whitelist above (default)
    #       always  — dump regardless of behaviour
    #       never   — skip regardless of behaviour
    individual_json_path = os.path.join(model_dir, 'test', f'{exp_id}.json')
    full_dump_behaviours = {'adaptive_gt', 'adaptive_pred', 'switching_pred'}
    save_mode = getattr(test_args, 'save_episode_dump', 'auto') or 'auto'
    if save_mode == 'always':
        should_dump = True
    elif save_mode == 'never':
        should_dump = False
    else:  # 'auto'
        should_dump = behaviour in full_dump_behaviours

    if should_dump:
        with open(individual_json_path, 'w') as f:
            json.dump(full_experiment_data, f, indent=4)
        logging.info(f"Full experiment data saved to {individual_json_path}")
    else:
        logging.info(
            f"Skipped per-episode dump for behaviour='{behaviour}' "
            f"(save_episode_dump={save_mode}). Summary still recorded in "
            f"all_evaluations.json + evaluation_data_scale_*.csv."
        )

    # 2. Update/Create summary index of ALL experiments
    summary_json_path = os.path.join(model_dir, 'test', 'all_evaluations.json')
    all_summaries = {}
    if os.path.exists(summary_json_path):
        try:
            with open(summary_json_path, 'r') as f:
                all_summaries = json.load(f)
        except Exception:
            all_summaries = {}

    # Create a lightweight entry for the master index
    summary_entry = full_experiment_data.copy()
    del summary_entry['episodes'] # Remove heavy data for the index
    summary_entry['data_file'] = f'{exp_id}.json'
    
    all_summaries[exp_id] = summary_entry
    
    with open(summary_json_path, 'w') as f:
        json.dump(all_summaries, f, indent=4)
    
    # Keep CSV for backward compatibility (per-run)
    csv_file_path = os.path.join(model_dir, 'test', f'evaluation_data_scale_{lora_scale}.csv')
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Success Times', 'Collision Times', 'Timeout Times', 'Path Length', 'Min Distance', 'Avg Uncertainty'])
        
        max_length = max(len(success_times), len(collision_times), len(timeout_times), len(all_path_len), len(min_dist), len(all_avg_uncertainty))
        for i in range(max_length):
            row = [
                success_times[i] if i < len(success_times) else '',
                collision_times[i] if i < len(collision_times) else '',
                timeout_times[i] if i < len(timeout_times) else '',
                all_path_len[i] if i < len(all_path_len) else '',
                min_dist[i] if i < len(min_dist) else '',
                all_avg_uncertainty[i] if i < len(all_avg_uncertainty) else ''
            ]
            writer.writerow(row)
    if total_awareness_predictions > 0:
        accuracy = (correct_awareness_predictions / total_awareness_predictions) * 100
        print(f"\n==========================================================")
        print(f"AWARENESS PREDICTION ACCURACY")
        print(f"==========================================================")
        print(f"Total Predictions: {total_awareness_predictions}")
        print(f"Correct Predictions: {correct_awareness_predictions}")
        
        # Show 0.5 threshold for network prediction, or the discrepancy threshold for other behaviors
        active_threshold = 0.5 if '_pred' in behaviour else getattr(test_args, 'discrepancy_threshold', 0.05)
        print(f"Accuracy: {accuracy:.2f}% (Threshold: {active_threshold}, M: {getattr(test_args, 'discrepancy_m', 1)})")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"==========================================================\n")

    # Save discrepancy data for analysis
    data_path = os.path.join(model_dir, 'test', f'discrepancy_data_{scenario}.json')
    with open(data_path, 'w') as f:
        json.dump(all_discrepancy_data, f, indent=4)
    logging.info(f"Discrepancy data saved to {data_path}")

    eval_envs.close()
"""
adaptive_predictive_mpc_demo.py

Minimal prototype of a predictive MPC crowd-navigation baseline inspired by:
  Le et al., "Social Navigation in Crowded Environments with Model Predictive
  Control and Deep Learning-Based Human Trajectory Prediction."

Purpose
-------
This is NOT a full reproduction of the paper. It is a simple plumbing test for:

    current human states
        -> constant-velocity human prediction
        -> awareness/reactivity-based kappa
        -> adaptive d_min
        -> predictive MPC
        -> execute first robot action

Humans move in straight lines with constant velocity.
No ORCA and no learned trajectory predictor are used.

Adaptive clearance
------------------
The requested rule is:

    d_min(t) = (1 - kappa_t) + d0

Thus:
    kappa = 0 -> d_min = d0 + 1.0 m  (most conservative)
    kappa = 1 -> d_min = d0          (most aggressive)

The demo computes kappa with the same spatial weighting idea used by the
adaptive policy:

    distance_weight  = max(0, 1 - distance / sensor_range)
    direction_weight = cos(theta / 2)
    w_i              = distance_weight * direction_weight

    kappa = sum_i w_i * I(awareness_i > 0.5) / sum_i w_i

If no observed human has nonzero weight, kappa defaults to 1.0.

IMPORTANT FOR THE REAL BASELINE
-------------------------------
1. Replace `predict_humans_constant_velocity(...)` with the EXISTING GST
   prediction already produced by the navigation stack. Do NOT run a second
   predictor inside the MPC baseline.

2. Replace the demo `Human.awareness` labels with the EXACT same kappa
   computation / awareness-estimator output used by the proposed LoRA method.
   The adaptive MPC and LoRA method should receive the same kappa_t.

3. Keep the same sensing radius and visibility/filtering as the proposed
   method. Do not give MPC privileged access to humans outside sensor range.

4. Before reporting Adaptive MPC, sweep FIXED d_min values to establish the
   MPC safety-efficiency curve and choose a sensible d0 / conservative range.

5. For a paper-quality implementation, consider parameterizing the CasADi NLP
   and warm-starting IPOPT rather than rebuilding the Opti problem every step.

Dependencies
------------
    pip install numpy casadi matplotlib
"""

from dataclasses import dataclass
import math
from typing import List, Tuple

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DT = 0.25               # simulator / control period [s]
HORIZON = 8             # MPC prediction horizon
SENSOR_RANGE = 5.0      # same idea as the paper setup [m]

V_MAX = 1.0             # robot max component-wise velocity [m/s]
A_MAX = 2.0             # robot max component-wise acceleration [m/s^2]

# Physical center-to-center collision threshold for this simple demo.
ROBOT_RADIUS = 0.30
HUMAN_RADIUS = 0.30
COLLISION_DISTANCE = ROBOT_RADIUS + HUMAN_RADIUS

# Adaptive d_min:
#     d_min = (1 - kappa) + D0
#
# D0 is the minimum clearance when kappa = 1 (most aggressive).
# 0.60 m corresponds to the sum of the demo robot/human radii.
D0 = COLLISION_DISTANCE

# Le-style MPC objective parameters.
# We keep the structure of goal + acceleration + jerk + soft collision cost.
W_GOAL = 10.0
W_ACCEL = 0.1
W_JERK = 0.1

# The paper uses a very large collision penalty. A smaller value is used here
# to make this toy optimization numerically friendlier. Tune for your setup.
W_COLL = 1.0e4

RHO = 0.5               # speed-dependent clearance term [s^2]
SOFTPLUS_MU = 20.0      # smoothness of soft collision penalty

MAX_SIM_STEPS = 240
GOAL_TOL = 0.30


# ---------------------------------------------------------------------------
# Simple human model
# ---------------------------------------------------------------------------

@dataclass
class Human:
    pos: np.ndarray
    vel: np.ndarray

    # Demo-only label in [0,1].
    # It does NOT change human motion here; all humans walk straight.
    # In the real baseline, replace this with your existing awareness /
    # reactivity estimator output or exact existing kappa pipeline.
    awareness: float


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def robot_heading(robot_pos: np.ndarray,
                  robot_vel: np.ndarray,
                  goal: np.ndarray) -> float:
    """Use velocity heading when moving; otherwise face the goal."""
    if np.linalg.norm(robot_vel) > 1e-4:
        return math.atan2(robot_vel[1], robot_vel[0])

    direction = goal - robot_pos
    return math.atan2(direction[1], direction[0])


# ---------------------------------------------------------------------------
# kappa_t from the adaptive policy's spatial weighting
# ---------------------------------------------------------------------------

def compute_kappa(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    goal: np.ndarray,
    humans: List[Human],
    sensor_range: float = SENSOR_RANGE,
    awareness_threshold: float = 0.5,
) -> Tuple[float, List[int]]:
    """
    Compute kappa from humans currently inside sensor range.

    Weight:
        w_i = max(0, 1 - dist/R) * cos(theta_i / 2)

    Binary reactivity:
        friendly_i = I(awareness_i > threshold)

    Aggregation:
        kappa = sum(w_i * friendly_i) / sum(w_i)

    No relevant human:
        kappa = 1.0
    """
    psi = robot_heading(robot_pos, robot_vel, goal)

    total_weight = 0.0
    friendly_weight = 0.0
    visible_ids = []

    for i, human in enumerate(humans):
        rel = human.pos - robot_pos
        dist = float(np.linalg.norm(rel))

        if dist > sensor_range:
            continue

        visible_ids.append(i)

        los_angle = math.atan2(rel[1], rel[0])
        theta = wrap_to_pi(los_angle - psi)

        distance_weight = max(0.0, 1.0 - dist / sensor_range)
        direction_weight = max(0.0, math.cos(theta / 2.0))
        weight = distance_weight * direction_weight

        if weight <= 0.0:
            continue

        is_friendly = 1.0 if human.awareness > awareness_threshold else 0.0

        total_weight += weight
        friendly_weight += weight * is_friendly

    if total_weight <= 1e-12:
        return 1.0, visible_ids

    kappa = friendly_weight / total_weight
    return float(np.clip(kappa, 0.0, 1.0)), visible_ids


def adaptive_dmin(kappa: float, d0: float = D0) -> float:
    """Requested adaptive clearance law."""
    return (1.0 - float(kappa)) + float(d0)


# ---------------------------------------------------------------------------
# Deterministic constant-velocity predictor
# ---------------------------------------------------------------------------

def predict_humans_constant_velocity(
    humans: List[Human],
    visible_ids: List[int],
    horizon: int = HORIZON,
    dt: float = DT,
) -> np.ndarray:
    """
    Return predictions with shape:
        [num_visible_humans, horizon, 2]

    Prediction:
        p_i(k) = p_i(t) + (k+1) * dt * v_i(t)

    REAL IMPLEMENTATION:
        Replace this function's output with your existing GST future positions.
    """
    predictions = []

    for idx in visible_ids:
        human = humans[idx]
        traj = []
        for k in range(horizon):
            future_t = (k + 1) * dt
            traj.append(human.pos + future_t * human.vel)
        predictions.append(traj)

    if len(predictions) == 0:
        return np.zeros((0, horizon, 2), dtype=float)

    return np.asarray(predictions, dtype=float)


# ---------------------------------------------------------------------------
# Predictive MPC
# ---------------------------------------------------------------------------

def stable_softplus(z, mu: float):
    """
    Smooth approximation of max(z, 0):
        log(1 + exp(mu*z)) / mu

    Written in a numerically safer piecewise form.
    """
    x = mu * z
    return ca.if_else(
        x > 0,
        (x + ca.log(1 + ca.exp(-x))) / mu,
        ca.log(1 + ca.exp(x)) / mu,
    )


def solve_mpc(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    goal: np.ndarray,
    human_predictions: np.ndarray,
    d_min: float,
    previous_accel: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Solve one Le-style receding-horizon MPC problem.

    Robot model:
        p_{k+1} = p_k + dt*v_k + 0.5*dt^2*a_k
        v_{k+1} = v_k + dt*a_k

    Cost:
        goal tracking + acceleration + jerk + soft collision penalty

    Only the FIRST acceleration is returned/executed.
    """
    opti = ca.Opti()

    # Decision variables.
    A = opti.variable(2, HORIZON)       # acceleration
    P = opti.variable(2, HORIZON + 1)  # position
    V = opti.variable(2, HORIZON + 1)  # velocity

    # Initial state.
    opti.subject_to(P[:, 0] == robot_pos)
    opti.subject_to(V[:, 0] == robot_vel)

    cost = 0

    # Straight-line reference progressing toward goal at up to V_MAX.
    ref = robot_pos.astype(float).copy()

    for k in range(HORIZON):
        # Double-integrator dynamics.
        p_next = P[:, k] + DT * V[:, k] + 0.5 * (DT ** 2) * A[:, k]
        v_next = V[:, k] + DT * A[:, k]

        opti.subject_to(P[:, k + 1] == p_next)
        opti.subject_to(V[:, k + 1] == v_next)

        # Component-wise bounds, matching the simple form in the paper.
        opti.subject_to(opti.bounded(-A_MAX, A[0, k], A_MAX))
        opti.subject_to(opti.bounded(-A_MAX, A[1, k], A_MAX))
        opti.subject_to(opti.bounded(-V_MAX, V[0, k + 1], V_MAX))
        opti.subject_to(opti.bounded(-V_MAX, V[1, k + 1], V_MAX))

        # Reference point: move toward the goal by V_MAX * DT.
        to_goal = goal - ref
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal > 1e-9:
            step = min(V_MAX * DT, dist_goal)
            ref = ref + step * to_goal / dist_goal

        ref_ca = ca.DM(ref)

        # Goal tracking.
        goal_error = P[:, k + 1] - ref_ca
        cost += W_GOAL * ca.dot(goal_error, goal_error)

        # Acceleration.
        cost += W_ACCEL * ca.dot(A[:, k], A[:, k])

        # Jerk.
        if k == 0:
            jerk = A[:, k] - ca.DM(previous_accel)
        else:
            jerk = A[:, k] - A[:, k - 1]
        cost += W_JERK * ca.dot(jerk, jerk)

        # Soft collision penalty against deterministic predicted human positions.
        for i in range(human_predictions.shape[0]):
            human_pos = ca.DM(human_predictions[i, k])

            delta = P[:, k + 1] - human_pos
            dist_sq = ca.dot(delta, delta)
            speed_sq = ca.dot(V[:, k + 1], V[:, k + 1])

            # Le-style speed-dependent safety expression:
            #
            #   ||p_robot - p_human||^2 >= d_min^2 + rho ||v_robot||^2
            #
            violation = (d_min ** 2) + RHO * speed_sq - dist_sq
            cost += W_COLL * stable_softplus(violation, SOFTPLUS_MU)

    opti.minimize(cost)

    # Initial guess: continue with current velocity / zero acceleration.
    opti.set_initial(A, 0.0)
    for k in range(HORIZON + 1):
        opti.set_initial(P[:, k], robot_pos + k * DT * robot_vel)
        opti.set_initial(V[:, k], robot_vel)

    opts = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 150,
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        accel = np.asarray(sol.value(A[:, 0]), dtype=float).reshape(2)
        accel = np.clip(accel, -A_MAX, A_MAX)
        return accel, True

    except RuntimeError:
        # Toy fallback: accelerate toward goal.
        desired_vel = goal - robot_pos
        norm = np.linalg.norm(desired_vel)
        if norm > 1e-9:
            desired_vel = V_MAX * desired_vel / norm
        else:
            desired_vel = np.zeros(2)

        accel = (desired_vel - robot_vel) / DT
        accel = np.clip(accel, -A_MAX, A_MAX)
        return accel, False


# ---------------------------------------------------------------------------
# Demo simulator
# ---------------------------------------------------------------------------

def make_demo_humans() -> List[Human]:
    """
    Straight-line crossing pedestrians.

    Awareness is ONLY used for kappa in this demo.
    Everyone still follows constant-velocity motion.
    """
    return [
        Human(np.array([-1.0, -3.5]), np.array([0.00, 0.55]), 0.0),
        Human(np.array([ 0.0,  3.5]), np.array([0.00,-0.55]), 1.0),
        Human(np.array([ 1.2, -3.5]), np.array([0.00, 0.50]), 1.0),
        Human(np.array([ 2.0,  3.5]), np.array([0.00,-0.45]), 0.0),
        Human(np.array([ 3.0, -3.0]), np.array([-0.15, 0.45]), 1.0),
        Human(np.array([-2.5,  2.5]), np.array([0.20,-0.35]), 0.0),
    ]


def simulate():
    robot_pos = np.array([-5.0, 0.0], dtype=float)
    robot_vel = np.zeros(2, dtype=float)
    goal = np.array([5.0, 0.0], dtype=float)

    humans = make_demo_humans()

    previous_accel = np.zeros(2)

    robot_history = [robot_pos.copy()]
    human_history = [[h.pos.copy() for h in humans]]
    kappa_history = []
    dmin_history = []

    reached_goal = False
    collision = False

    for step in range(MAX_SIM_STEPS):
        # ---------------------------------------------------------------
        # 1. Compute the SAME kappa_t concept used by the adaptive policy.
        # ---------------------------------------------------------------
        kappa, visible_ids = compute_kappa(
            robot_pos=robot_pos,
            robot_vel=robot_vel,
            goal=goal,
            humans=humans,
            sensor_range=SENSOR_RANGE,
        )

        # ---------------------------------------------------------------
        # 2. Convert kappa_t into adaptive MPC clearance.
        #
        #        d_min = (1 - kappa) + d0
        # ---------------------------------------------------------------
        d_min = adaptive_dmin(kappa, D0)

        # ---------------------------------------------------------------
        # 3. Deterministic prediction.
        #
        # REAL CODE: replace this with existing GST prediction output.
        # ---------------------------------------------------------------
        human_predictions = predict_humans_constant_velocity(
            humans,
            visible_ids,
            horizon=HORIZON,
            dt=DT,
        )

        # ---------------------------------------------------------------
        # 4. Solve MPC and execute only first control.
        # ---------------------------------------------------------------
        accel, solved = solve_mpc(
            robot_pos=robot_pos,
            robot_vel=robot_vel,
            goal=goal,
            human_predictions=human_predictions,
            d_min=d_min,
            previous_accel=previous_accel,
        )

        # Simulator update.
        robot_pos = robot_pos + DT * robot_vel + 0.5 * (DT ** 2) * accel
        robot_vel = robot_vel + DT * accel
        robot_vel = np.clip(robot_vel, -V_MAX, V_MAX)
        previous_accel = accel

        # Humans: straight-line constant velocity, no ORCA.
        for human in humans:
            human.pos = human.pos + DT * human.vel

        robot_history.append(robot_pos.copy())
        human_history.append([h.pos.copy() for h in humans])
        kappa_history.append(kappa)
        dmin_history.append(d_min)

        # Physical collision check (independent of adaptive d_min).
        min_human_dist = min(
            np.linalg.norm(robot_pos - h.pos) for h in humans
        )
        if min_human_dist < COLLISION_DISTANCE:
            collision = True
            print(
                f"[step {step:03d}] COLLISION "
                f"dist={min_human_dist:.3f} m "
                f"kappa={kappa:.3f} d_min={d_min:.3f}"
            )
            break

        goal_dist = np.linalg.norm(goal - robot_pos)

        if step % 10 == 0:
            print(
                f"[step {step:03d}] "
                f"goal_dist={goal_dist:.2f} m "
                f"visible={len(visible_ids)} "
                f"kappa={kappa:.3f} "
                f"d_min={d_min:.3f} m "
                f"solver={'OK' if solved else 'FALLBACK'}"
            )

        if goal_dist < GOAL_TOL:
            reached_goal = True
            break

    robot_history = np.asarray(robot_history)
    human_history = np.asarray(human_history)

    print("\n--- RESULT ---")
    print(f"Reached goal : {reached_goal}")
    print(f"Collision    : {collision}")
    print(f"Steps        : {len(robot_history) - 1}")
    print(f"Time         : {(len(robot_history) - 1) * DT:.2f} s")

    if kappa_history:
        print(f"Mean kappa   : {np.mean(kappa_history):.3f}")
        print(f"Mean d_min   : {np.mean(dmin_history):.3f} m")
        print(f"d_min range  : [{np.min(dmin_history):.3f}, "
              f"{np.max(dmin_history):.3f}] m")

    # One simple trajectory figure.
    plt.figure(figsize=(8, 7))
    plt.plot(robot_history[:, 0], robot_history[:, 1],
             linewidth=2.5, label="Robot")
    plt.scatter(robot_history[0, 0], robot_history[0, 1],
                marker="o", s=80, label="Robot start")
    plt.scatter(goal[0], goal[1], marker="*", s=180, label="Goal")

    for i in range(human_history.shape[1]):
        label = f"Human {i} (aware={humans[i].awareness:.0f})"
        plt.plot(human_history[:, i, 0], human_history[:, i, 1],
                 linewidth=1.2, label=label)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Adaptive Predictive MPC Toy Simulator")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate()

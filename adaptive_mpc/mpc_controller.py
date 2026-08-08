from dataclasses import dataclass
import time

import numpy as np


@dataclass
class MPCConfig:
    dt: float
    horizon: int
    v_max: float
    a_max: float = 2.0
    w_goal: float = 10.0
    w_accel: float = 0.1
    w_jerk: float = 0.1
    w_coll: float = 1.0e4
    rho: float = 0.5
    softplus_mu: float = 20.0


@dataclass
class MPCResult:
    velocity: np.ndarray
    accel: np.ndarray
    solved: bool
    solve_time_ms: float


def _stable_softplus(ca, z, mu):
    x = mu * z
    return ca.if_else(
        x > 0,
        (x + ca.log(1 + ca.exp(-x))) / mu,
        ca.log(1 + ca.exp(x)) / mu,
    )


class AdaptiveMPCController:
    def __init__(self, config):
        self.config = config
        self.previous_accel = np.zeros(2, dtype=float)
        self.previous_feasible_velocity = np.zeros(2, dtype=float)

    def reset(self, initial_velocity):
        self.previous_accel = np.zeros(2, dtype=float)
        self.previous_feasible_velocity = np.asarray(initial_velocity, dtype=float).reshape(2)
        self.previous_feasible_velocity = self._clip_velocity(self.previous_feasible_velocity)

    def _clip_velocity(self, velocity):
        velocity = np.asarray(velocity, dtype=float).reshape(2)
        norm = float(np.linalg.norm(velocity))
        if norm > self.config.v_max and norm > 1e-12:
            velocity = velocity / norm * self.config.v_max
        return velocity

    def solve(self, robot_pos, robot_vel, goal, human_predictions, d_min):
        try:
            import casadi as ca
        except ImportError as exc:
            raise ImportError(
                "Adaptive MPC requires casadi. Rebuild the Docker image after "
                "the Dockerfile dependency update, or install casadi in the container."
            ) from exc

        robot_pos = np.asarray(robot_pos, dtype=float).reshape(2)
        robot_vel = np.asarray(robot_vel, dtype=float).reshape(2)
        goal = np.asarray(goal, dtype=float).reshape(2)
        human_predictions = np.asarray(human_predictions, dtype=float)
        if human_predictions.size == 0:
            human_predictions = np.zeros((0, self.config.horizon, 2), dtype=float)
        human_predictions = human_predictions[:, :self.config.horizon, :]

        t0 = time.perf_counter()
        opti = ca.Opti()
        a = opti.variable(2, self.config.horizon)
        p = opti.variable(2, self.config.horizon + 1)
        v = opti.variable(2, self.config.horizon + 1)

        opti.subject_to(p[:, 0] == robot_pos)
        opti.subject_to(v[:, 0] == robot_vel)

        cost = 0
        ref = robot_pos.copy()
        dt = self.config.dt

        for k in range(self.config.horizon):
            p_next = p[:, k] + dt * v[:, k] + 0.5 * (dt ** 2) * a[:, k]
            v_next = v[:, k] + dt * a[:, k]
            opti.subject_to(p[:, k + 1] == p_next)
            opti.subject_to(v[:, k + 1] == v_next)
            opti.subject_to(opti.bounded(-self.config.a_max, a[0, k], self.config.a_max))
            opti.subject_to(opti.bounded(-self.config.a_max, a[1, k], self.config.a_max))
            opti.subject_to(opti.bounded(-self.config.v_max, v[0, k + 1], self.config.v_max))
            opti.subject_to(opti.bounded(-self.config.v_max, v[1, k + 1], self.config.v_max))

            to_goal = goal - ref
            dist_goal = float(np.linalg.norm(to_goal))
            if dist_goal > 1e-9:
                step = min(self.config.v_max * dt, dist_goal)
                ref = ref + step * to_goal / dist_goal

            goal_error = p[:, k + 1] - ca.DM(ref)
            cost += self.config.w_goal * ca.dot(goal_error, goal_error)
            cost += self.config.w_accel * ca.dot(a[:, k], a[:, k])

            if k == 0:
                jerk = a[:, k] - ca.DM(self.previous_accel)
            else:
                jerk = a[:, k] - a[:, k - 1]
            cost += self.config.w_jerk * ca.dot(jerk, jerk)

            for i in range(human_predictions.shape[0]):
                human_pos = ca.DM(human_predictions[i, k])
                delta = p[:, k + 1] - human_pos
                dist_sq = ca.dot(delta, delta)
                speed_sq = ca.dot(v[:, k + 1], v[:, k + 1])
                violation = (float(d_min) ** 2) + self.config.rho * speed_sq - dist_sq
                cost += self.config.w_coll * _stable_softplus(ca, violation, self.config.softplus_mu)

        opti.minimize(cost)
        opti.set_initial(a, 0.0)
        for k in range(self.config.horizon + 1):
            opti.set_initial(p[:, k], robot_pos + k * dt * robot_vel)
            opti.set_initial(v[:, k], robot_vel)

        opts = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 150,
        }
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            accel = np.asarray(sol.value(a[:, 0]), dtype=float).reshape(2)
            accel = np.clip(accel, -self.config.a_max, self.config.a_max)
            velocity = self._clip_velocity(robot_vel + dt * accel)
            self.previous_accel = accel
            self.previous_feasible_velocity = velocity
            solved = True
        except RuntimeError:
            accel = np.zeros(2, dtype=float)
            velocity = self.previous_feasible_velocity.copy()
            solved = False

        solve_time_ms = (time.perf_counter() - t0) * 1000.0
        return MPCResult(velocity=velocity, accel=accel, solved=solved, solve_time_ms=solve_time_ms)

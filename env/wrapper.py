"""ir-sim env wrapper: turns a world YAML into an RL training/eval interface.

Observation (obs, float32 vector), input modeled on prepare_state from DRL-robot-navigation-IR-SIM:
    [lidar binned-min values / range_max (max_bins,),
     dist / goal_dist_norm, cos, sin,
     previous chunk action normalized (action_dim*prev_chunk_size,) (optional)]
Action (numpy, (action_dim,)):
    normalized to [-1,1] network output, internally mapped back to the real velocity range [vel_min, vel_max]
    (diff-drive robot: [linear_vel, angular_vel]).

lidar binning (aligned with DRL prepare_state):
    - bin_size = ceil(lidar_len / max_bins), take the min value per bin then / range_max.
    - when max_bins = lidar_len, bin_size=1, equivalent to per-beam / range_max.

Previous chunk action (action-chunk adapted version):
    - in chunk mode the model runs once per decision point (every N steps), and the observation encodes
      "the chunk that was just executed", rather than the previous single-step action.
    - per-dim DRL normalization: each step [lin*2, (ang+1)/2].
    - episode start uses all zeros; when the chunk ends early (done), pad with zeros to a fixed length.
"""

from __future__ import annotations

import os
import sys

# irsim calls matplotlib.use("TkAgg"/"Qt5Agg") explicitly at import time (BACKEND_PREFERENCES in
# irsim.env.env_base); on headless servers it prints a bunch of "Failed to use ... backend" warnings
# before falling back to Agg (the MPLBACKEND env var does not affect an explicit use()). Here we intercept
# matplotlib.use before irsim: silently fall back to Agg when the GUI backend fails to load; on desktop/Windows
# the use() succeeds and behavior is unchanged.
if "matplotlib" not in sys.modules:
    try:
        import matplotlib as _mpl

        _real_use = _mpl.use

        def _silent_fallback_use(backend, *args, **kwargs):
            try:
                return _real_use(backend, *args, **kwargs)
            except Exception:
                return _real_use("Agg", *args, **kwargs)

        _mpl.use = _silent_fallback_use
    except ImportError:
        pass

import numpy as np
import irsim
from irsim.util.random import rng

from .reward import RewardFn, wrap_angle


class IrSimEnv:
    """Wrap :class:`irsim.env.EnvBase` as an RL environment.

    Args:
        world_name: world YAML path (absolute or relative to the repo root).
        reward_cfg: reward config section.
        obs_cfg: obs config section (lidar_range_max etc.).
        env_cfg: env config section (random_start / start_range / fixed_goal etc.).
        seed: ir-sim random seed.
        display: whether to show visualization (False during training).
        log_level: ir-sim log level.
    """

    def __init__(
        self,
        world_name: str,
        reward_cfg: dict,
        obs_cfg: dict,
        env_cfg: dict | None = None,
        seed: int | None = None,
        display: bool = False,
        log_level: str = "WARNING",
    ) -> None:
        self._env = irsim.make(
            world_name,
            display=display,
            disable_all_plot=not display,
            log_level=log_level,
            seed=seed,
        )

        # Override the simulation step (env_cfg.step_time, e.g. 0.0833 to match the real 12Hz lidar frame rate).
        # ir-sim physics integration uses _world_param.step_time, while clock/display use _world.step_time; both must be changed.
        env_cfg = env_cfg or {}
        if env_cfg.get("step_time") is not None:
            st = float(env_cfg["step_time"])
            self._env._world.step_time = st
            self._env._world_param.step_time = st

        self.display = display
        self.render_interval = float(obs_cfg.get("render_interval", self._env.step_time))
        self.lidar_range_max = float(obs_cfg.get("lidar_range_max", 10.0))
        self.goal_dist_norm = float(obs_cfg.get("goal_dist_norm", 10.0))
        self.include_goal = bool(obs_cfg.get("include_goal", True))
        self.goal_dim = int(obs_cfg.get("goal_dim", 2))
        self.max_steps = int(reward_cfg.get("max_steps", 500))

        self.random_start = bool(env_cfg.get("random_start", False))
        self.start_range_low = np.asarray(
            env_cfg.get("start_range_low", [0.5, 0.5, -3.14]), dtype=float
        )
        self.start_range_high = np.asarray(
            env_cfg.get("start_range_high", [9.5, 9.5, 3.14]), dtype=float
        )
        self.fixed_goal = env_cfg.get("fixed_goal", None)

        self.action_dim = int(obs_cfg.get("action_dim", 2))
        self.vel_min = np.array(obs_cfg.get("vel_min", [-1.0, -1.0]), dtype=np.float32)
        self.vel_max = np.array(obs_cfg.get("vel_max", [1.0, 1.0]), dtype=np.float32)

        self.include_prev_action = bool(obs_cfg.get("include_prev_action", False))
        self.prev_chunk_size = int(obs_cfg.get("prev_chunk_size", 0))
        self.prev_action_dim = (
            self.action_dim * self.prev_chunk_size if self.include_prev_action else 0
        )
        self._prev_chunk = np.zeros(self.prev_action_dim, dtype=np.float32)

        self.reward_fn = RewardFn(reward_cfg)
        self.step_time = float(self._env.step_time)

        scan = self._env.get_lidar_scan()
        self.lidar_len = int(np.asarray(scan["ranges"]).size)

        self.max_bins = int(obs_cfg.get("max_bins", self.lidar_len))
        self.obs_dim = (
            self.max_bins + (self.goal_dim if self.include_goal else 0) + self.prev_action_dim
        )
        self.step_count = 0
        self._prev_dist: float | None = None

    @property
    def robot(self):
        """The current robot object (rebuilt after reset(random=True); must be fetched dynamically)."""
        return self._env.robot

    @property
    def env(self):
        return self._env

    # ------------------------------------------------------------------ #
    # Internal state/observation reads
    # ------------------------------------------------------------------ #
    def _dist_to_goal(self) -> float:
        state = np.squeeze(np.asarray(self._env.get_robot_state(), dtype=np.float64))
        goal = np.squeeze(np.asarray(self.robot.goal, dtype=np.float64))
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        return float(np.hypot(dx, dy))

    def _angle_to_goal(self) -> float:
        state = np.squeeze(np.asarray(self._env.get_robot_state(), dtype=np.float64))
        goal = np.squeeze(np.asarray(self.robot.goal, dtype=np.float64))
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        return wrap_angle(float(np.arctan2(dy, dx)) - float(state[2]))

    def _goal_obs(self) -> np.ndarray:
        state = np.squeeze(np.asarray(self._env.get_robot_state(), dtype=np.float64))
        goal = np.squeeze(np.asarray(self.robot.goal, dtype=np.float64))
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        dist = float(np.hypot(dx, dy))
        angle = self._angle_to_goal()
        return np.array(
            [dist / self.goal_dist_norm, np.cos(angle), np.sin(angle)],
            dtype=np.float32,
        )[: self.goal_dim]

    def _bin_lidar(self, ranges: np.ndarray) -> np.ndarray:
        """DRL-style lidar binning: take the min per bin_size beams then / range_max.

        Args:
            ranges: raw lidar ranges (inf treated as range_max).

        Returns:
            (max_bins,) float32, values in [0, 1].
        """
        ranges = np.where(np.isinf(ranges), self.lidar_range_max, ranges)
        ranges = np.clip(ranges, 0.0, self.lidar_range_max)
        bin_size = int(np.ceil(len(ranges) / self.max_bins))
        min_values = []
        for i in range(0, len(ranges), bin_size):
            bin = ranges[i : i + min(bin_size, len(ranges) - i)]
            min_values.append(min(bin))
        # Pad to max_bins (if lidar_len is not an integer multiple of max_bins, fill the tail with full range = unobstructed)
        while len(min_values) < self.max_bins:
            min_values.append(self.lidar_range_max)
        return (np.asarray(min_values, dtype=np.float32) / self.lidar_range_max)[: self.max_bins]

    def _get_obs(self) -> np.ndarray:
        ranges = np.asarray(self._env.get_lidar_scan()["ranges"], dtype=np.float64)
        obs = [self._bin_lidar(ranges)]
        if self.include_goal:
            obs.append(self._goal_obs())
        if self.include_prev_action:
            obs.append(self._prev_chunk)
        return np.concatenate(obs).astype(np.float32)

    def _set_prev_chunk(self, executed: np.ndarray) -> None:
        """Fixed-length encode the just-executed actions (possibly truncated) as the previous-chunk observation.

        Per-dim DRL normalization: each step [lin*2, (ang+1)/2] (action already normalized to [-1,1]).
        """
        if not self.include_prev_action:
            return
        a = np.asarray(executed, dtype=np.float32).reshape(-1, self.action_dim)
        n = self.prev_action_dim
        chunk = np.zeros(n, dtype=np.float32)
        for k, row in enumerate(a):
            if k >= self.prev_chunk_size:
                break
            lin = row[0] * 2.0
            ang = (row[1] + 1.0) / 2.0
            chunk[k * self.action_dim : (k + 1) * self.action_dim] = [lin, ang]
        self._prev_chunk = chunk

    def _robot_status(self) -> tuple[bool, bool]:
        collision = bool(self.robot.collision)
        arrive = bool(self.robot.arrive)
        return collision, arrive

    # ------------------------------------------------------------------ #
    # Action mapping
    # ------------------------------------------------------------------ #
    def scale_action(self, action_norm: np.ndarray) -> np.ndarray:
        """Network output [-1,1] -> real velocity."""
        a = np.clip(np.asarray(action_norm, dtype=np.float64), -1.0, 1.0)
        return self.vel_min + (a + 1.0) / 2.0 * (self.vel_max - self.vel_min)

    # ------------------------------------------------------------------ #
    # RL interface
    # ------------------------------------------------------------------ #
    def _maybe_randomize_start(self) -> None:
        """Randomize the robot's initial pose (keep the goal fixed).

        Called after reset(random=True) rebuilds the scene: resample a start pose
        that does not overlap any obstacle, and pin the goal to the configured value
        (the world YAML's goal can also stay fixed).
        """
        if not self.random_start:
            return
        if self.fixed_goal is not None:
            self.robot.set_goal(list(self.fixed_goal))
        obstacles = [o for o in self._env.objects if o is not self.robot]
        for _ in range(50):
            pose = rng.uniform(self.start_range_low, self.start_range_high)
            self.robot.set_state(pose)
            if not any(self.robot.geometry.intersects(o.geometry) for o in obstacles):
                break
        self._env.refresh()

    def reset(self, random: bool = True) -> np.ndarray:
        """Reset the environment and return the initial observation (previous chunk action zeroed)."""
        self._env.reset(random=random)
        self._maybe_randomize_start()
        self.step_count = 0
        self.reward_fn.reset()
        self._prev_dist = None
        self._prev_chunk = np.zeros(self.prev_action_dim, dtype=np.float32)
        return self._get_obs()

    def step_single(self, action_norm: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Execute a single control step.

        Args:
            action_norm: normalized action (action_dim,).

        Returns:
            (obs, reward, done, info) consistent with gym.
        """
        action = self.scale_action(action_norm)
        self._env.step(np.asarray(action, dtype=np.float64).reshape(-1, 1), action_id=0)
        if self.display:
            self._env.render(interval=self.render_interval)
        self.step_count += 1

        obs = self._get_obs()
        collision, arrive = self._robot_status()
        timeout = self.step_count >= self.max_steps
        done = bool(arrive or collision or timeout)

        dist = self._dist_to_goal()
        laser_scan = np.asarray(self._env.get_lidar_scan()["ranges"], dtype=np.float64)
        reward = self.reward_fn(
            dist,
            collision,
            arrive,
            self._angle_to_goal(),
            action=action,
            laser_scan=laser_scan,
        )
        self._prev_dist = dist

        info = {
            "collision": collision,
            "arrive": arrive,
            "timeout": timeout,
            "dist_to_goal": dist,
            "step": self.step_count,
            "time": float(self._env.time),
        }
        return obs, reward, done, info

    def step_chunk(
        self, actions: np.ndarray, gamma: float = 0.99
    ) -> tuple[np.ndarray, float, bool, dict]:
        """Execute an action chunk open-loop (length N).

        Args:
            actions: normalized action array (N, action_dim).
            gamma: discount factor for the within-chunk sum.

        Returns:
            (next_obs, discounted_chunk_reward, done, info).
            Truncates early if done occurs mid-chunk (reached/collision/timeout).
        """
        rewards: list[float] = []
        info: dict = {}
        done = False
        executed: list[np.ndarray] = []
        for step, a in enumerate(np.asarray(actions).reshape(-1, self.action_dim)):
            obs, reward, done, info = self.step_single(a)
            executed.append(np.asarray(a, dtype=np.float32))
            rewards.append(reward)
            if done:
                break
        chunk_reward = sum(
            (gamma**k) * r for k, r in enumerate(rewards)
        )
        # Decision-point observation: state is the chunk final state, previous chunk action is the just-executed action
        self._set_prev_chunk(executed)
        obs = self._get_obs()
        return obs, float(chunk_reward), done, info

    def close(self) -> None:
        """Close the environment and release its matplotlib figure.

        Note: ``EnvBase.end()`` returns early when ``disable_all_plot=True`` (headless training)
        without calling ``plt.close``, so figures only accumulate (a warning triggers above 20).
        Here we add one explicit close targeting this environment's figure.
        """
        try:
            self._env.close()
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt

            env_plot = getattr(self._env, "_env_plot", None)
            if env_plot is not None:
                plt.close(env_plot.fig)
        except Exception:
            pass

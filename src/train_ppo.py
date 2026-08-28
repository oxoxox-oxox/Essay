"""SB3 PPO single-step (N=1) training script (ir-sim env).

Separated from src/eval_ppo.py: this file only does training (including the
periodic eval by EvalCallback during training and best_model saving);
use src/eval_ppo.py for the detailed evaluation after training.

Usage:
    python src/train_ppo.py --steps 300000 --device cuda   # full training
    python src/train_ppo.py --steps 20000 --device cpu      # smoke test

Artifacts:
    checkpoints/{name}_N1/best_model.zip   # eval-best saved by EvalCallback
    checkpoints/{name}_N1/ppo_model.zip    # final model at the end of training
    runs/{name}_N1/evaluations.npz         # eval curve (eval/success_rate etc.)
    runs/{name}_N1/tensorboard/            # TB logs
"""

from __future__ import annotations

import argparse
import os
import sys

# headless servers have no GUI: force the matplotlib Agg backend (avoids TkAgg/Qt5Agg load-failure warning spam).
# Training is always headless (display=False), set unconditionally here; respect MPLBACKEND if externally set.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, sync_envs_normalization

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.multi_world import MultiWorldIrSimEnv  # noqa: E402
from env.ppo_gym import PPOGymEnv  # noqa: E402
from env.wrapper import IrSimEnv  # noqa: E402
from utils.config import deep_update, load_config, resolve_path  # noqa: E402


class SuccessRateEvalCallback(EvalCallback):
    """EvalCallback that selects best_model by eval success rate (not mean reward).

    The reward here includes goal shaping, so a bad model may loiter in episodes
    farming shaping reward, making mean reward anti-correlated with success
    (high reward but low or even 0% success). SB3's default best_model selection
    by mean reward picks wrong models; this one selects by success_rate,
    using mean_reward as a tie-break when success is equal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_success_rate = -np.inf

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            success_rate = -np.inf
            if len(self._is_success_buffer) > 0:
                success_rate = float(np.mean(self._is_success_buffer))
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # Key change: select best by success_rate (mean_reward as tie-break when success is equal)
            is_best = success_rate > self.best_success_rate or (
                success_rate == self.best_success_rate
                and mean_reward > self.best_mean_reward
            )
            if is_best:
                if self.verbose >= 1:
                    print(f"New best success rate! ({100 * success_rate:.2f}%)")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_success_rate = success_rate
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


def make_env(cfg: dict, world_key: str, seed: int, display: bool = False):
    """Return a factory that builds a PPOGymEnv (wrapped in Monitor) for DummyVecEnv.

    Monitor purpose:
        1) removes the "not wrapped with Monitor" warning from evaluate_policy;
        2) records rollout/ep_rew_mean / ep_len_mean in TensorBoard during training.
    The env chain here has no reward-modifying wrapper, so Monitor does not change training semantics.

    Training env (world_key="world_name"): if env.worlds is configured, use the multi-world wrapper
    (random world per reset, single active scene, avoiding ir-sim global-state conflicts),
    otherwise use the single world_name. The eval env (world_key="eval_world") is always single-world.
    """

    def _factory():
        worlds = cfg["env"].get("worlds") or []
        if world_key == "world_name" and worlds:
            entries = [dict(w) if isinstance(w, dict) else {"world": w} for w in worlds]
            irsim_env = MultiWorldIrSimEnv(
                entries,
                reward_cfg=cfg["reward"],
                obs_cfg=cfg["obs"],
                env_cfg=cfg["env"],
                seed=seed,
                display=display,
                log_level="CRITICAL",
            )
        else:
            irsim_env = IrSimEnv(
                resolve_path(cfg["env"][world_key]),
                reward_cfg=cfg["reward"],
                obs_cfg=cfg["obs"],
                env_cfg=cfg["env"],
                seed=seed,
                display=display,
                log_level="CRITICAL",
            )
        return Monitor(PPOGymEnv(irsim_env, chunk_size=int(cfg["chunk"].get("size", 1))))

    return _factory


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SB3 PPO single-step baseline training (evaluate with src/eval_ppo.py)"
    )
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--name", default="ppo")
    ap.add_argument("--steps", type=int, default=300_000, help="total env steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-freq", type=int, default=20_000, help="training-time eval interval (env steps)")
    ap.add_argument("--eval-episodes", type=int, default=10, help="episodes per training-time eval")
    ap.add_argument("--hidden", type=str, default=None,
                    help="override hidden layers, e.g. '1024,1024' (default from config model.hidden1/2)")
    ap.add_argument("--worlds", type=str, default=None,
                    help="multi-world training list (comma-separated world YAML paths, overrides env.worlds in configs/train.yaml; "
                         "if not given, uses the config's env.worlds, and single-world training when the config is empty)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="action chunk length N (default from config chunk.size; 1=single-step, 5=output 5 steps at once)")
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="PPO entropy regularization coefficient (default 0.0; try 0.01 on entropy collapse)")
    ap.add_argument("--n-steps", type=int, default=1024,
                    help="PPO rollout steps (default 1024; with the 83ms step size, 4096 is recommended to cover more episodes)")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    # action chunk length N: network action space = N*action_dim; obs prev-action channel = N*action_dim
    chunk_size = args.chunk if args.chunk is not None else int(cfg["chunk"].get("size", 1))
    cfg = deep_update(cfg, {"obs": {"prev_chunk_size": chunk_size}, "chunk": {"size": chunk_size}})
    if args.worlds:
        # --worlds overrides the config's env.worlds (multi-world training)
        cfg = deep_update(cfg, {"env": {"worlds": [w.strip() for w in args.worlds.split(",")]}})

    if args.hidden:
        h1, h2 = (int(x) for x in args.hidden.split(","))
    else:
        h1 = int(cfg["model"].get("hidden1", 1024))
        h2 = int(cfg["model"].get("hidden2", 1024))

    tag = f"{args.name}_N{chunk_size}"
    checkpoint_dir = resolve_path(os.path.join(cfg["project"]["checkpoint_dir"], tag))
    run_dir = resolve_path(os.path.join(cfg["project"]["run_dir"], tag))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    train_env = DummyVecEnv([make_env(cfg, "world_name", args.seed)])
    eval_env = DummyVecEnv([make_env(cfg, "eval_world", args.seed + 1000)])

    eval_callback = SuccessRateEvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs={"net_arch": [h1, h2]},
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        device=args.device,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
        verbose=1,
    )
    print(f"[ppo] tag={tag} obs_dim={train_env.observation_space.shape} "
          f"act_dim={train_env.action_space.shape} net=[{h1},{h2}] "
          f"steps={args.steps} device={args.device}")

    model.learn(total_timesteps=args.steps, callback=eval_callback)

    final_path = os.path.join(checkpoint_dir, "ppo_model.zip")
    model.save(final_path)
    print(f"[ppo] final model saved: {final_path}")
    print(f"[ppo] training finished. For detailed evaluation run: "
          f"python src/eval_ppo.py --checkpoint {final_path}")


if __name__ == "__main__":
    main()

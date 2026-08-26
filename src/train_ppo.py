"""SB3 PPO 单步（N=1）训练脚本（ir-sim 环境）。

与 src/eval_ppo.py 分离：本文件只负责训练（含训练期 EvalCallback 的周期评估
与 best_model 保存）；训练结束后的详细评估请用 src/eval_ppo.py。

用法:
    python src/train_ppo.py --steps 300000 --device cuda   # 完整训练
    python src/train_ppo.py --steps 20000 --device cpu      # 冒烟测试

产物:
    checkpoints/{name}_N1/best_model.zip   # EvalCallback 保存的评估最优
    checkpoints/{name}_N1/ppo_model.zip    # 训练结束最终模型
    runs/{name}_N1/evaluations.npz         # 评估曲线（eval/success_rate 等）
    runs/{name}_N1/tensorboard/            # TB 日志
"""

from __future__ import annotations

import argparse
import os
import sys

# headless 服务器无 GUI：强制 matplotlib Agg 后端（避免 TkAgg/Qt5Agg 加载失败告警刷屏）。
# 训练恒为 headless（display=False），此处无条件设置；若外部已设 MPLBACKEND 环境变量则尊重它。
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, sync_envs_normalization

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.multi_world import MultiWorldIrSimEnv  # noqa: E402
from env.policy import GlobalCriticActorCriticPolicy  # noqa: E402
from env.ppo_gym import PPOGymEnv  # noqa: E402
from env.wrapper import IrSimEnv  # noqa: E402
from utils.config import deep_update, load_config, resolve_path  # noqa: E402


class SuccessRateEvalCallback(EvalCallback):
    """按 eval 成功率（而非 mean reward）选 best_model 的 EvalCallback。

    本任务的奖励含 goal shaping，坏模型会在 episode 里徘徊刷 shaping 奖励，
    导致 mean reward 与 success 反相关（reward 高但 success 低甚至 0%）。
    SB3 默认按 mean reward 存 best_model 会选错，这里改成按 success_rate 选，
    success 相同时用 mean_reward 做 tie-break。
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

            # 关键改动：按 success_rate 选 best（success 相同时用 mean_reward 做 tie-break）
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
    """返回一个构建 PPOGymEnv（外裹 Monitor）的工厂（DummyVecEnv 用）。

    Monitor 用途：
        1) 消除 evaluate_policy 的 "not wrapped with Monitor" 告警；
        2) 训练期 TensorBoard 记录 rollout/ep_rew_mean / ep_len_mean。
    本仓库环境链无任何修改 reward 的 wrapper，Monitor 不改变训练语义。

    训练环境（world_key="world_name"）：若配置了 env.worlds 则用多世界包装
    （每次 reset 随机换世界、单活跃场景，规避 ir-sim 全局状态冲突），
    否则用单个 world_name。评估环境（world_key="eval_world"）恒为单世界。
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
        description="SB3 PPO 单步基线训练（评估请用 src/eval_ppo.py）"
    )
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--name", default="ppo")
    ap.add_argument("--steps", type=int, default=300_000, help="总环境步数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-freq", type=int, default=20_000, help="训练期评估间隔（环境步）")
    ap.add_argument("--eval-episodes", type=int, default=10, help="训练期每次评估的 episode 数")
    ap.add_argument("--hidden", type=str, default=None,
                    help="覆盖隐藏层，如 '1024,1024'（默认取 config model.hidden1/2）")
    ap.add_argument("--worlds", type=str, default=None,
                    help="多世界训练列表（逗号分隔的世界 YAML 路径，覆盖 configs/train.yaml 的 env.worlds；"
                         "不传则用 config 里的 env.worlds，配置为空则单世界训练）")
    ap.add_argument("--chunk", type=int, default=None,
                    help="action chunk 长度 N（默认取 config chunk.size；1=单步，5=一次输出 5 步）")
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="PPO 熵正则系数（默认 0.0；熵坍缩时试 0.01）")
    ap.add_argument("--n-steps", type=int, default=1024,
                    help="PPO rollout 步数（默认 1024；83ms 步长下建议 4096 覆盖更多集）")
    ap.add_argument("--critic-global", action="store_true",
                    help="方向2：给 Critic 追加全局特征（绝对位姿+绝对目标，obs 尾部 6 维），"
                         "actor 保持局部观测（输入仍为 105 维，导出/部署链路不变）")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    # action chunk 长度 N：网络动作空间 = N*action_dim；obs 上一动作通道 = N*action_dim
    chunk_size = args.chunk if args.chunk is not None else int(cfg["chunk"].get("size", 1))
    cfg = deep_update(cfg, {"obs": {"prev_chunk_size": chunk_size}, "chunk": {"size": chunk_size}})
    if args.critic_global:
        cfg = deep_update(cfg, {"obs": {"include_global": True}})
    if args.worlds:
        # --worlds 覆盖 config 的 env.worlds（多世界训练）
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

    # 方向2：局部/全局 obs 维数（与 env/wrapper.IrSimEnv 的 obs_dim 计算保持一致）
    obs_cfg = cfg["obs"]
    local_obs_dim = (
        int(obs_cfg.get("max_bins", 100))
        + (int(obs_cfg.get("goal_dim", 2)) if obs_cfg.get("include_goal", True) else 0)
        + (int(obs_cfg.get("action_dim", 2)) * chunk_size if obs_cfg.get("include_prev_action", False) else 0)
    )
    global_dim = 6 if obs_cfg.get("include_global", False) else 0

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

    policy_cls: str | type = "MlpPolicy"
    policy_kwargs: dict = {"net_arch": [h1, h2]}
    if global_dim > 0:
        # 方向2：Critic 独享全局特征；actor 输入仍为 local_obs_dim
        policy_cls = GlobalCriticActorCriticPolicy
        policy_kwargs["local_obs_dim"] = local_obs_dim
        assert train_env.observation_space.shape[0] == local_obs_dim + global_dim, (
            f"obs 维度不一致: env={train_env.observation_space.shape[0]} "
            f"local+global={local_obs_dim}+{global_dim}"
        )

    model = PPO(
        policy_cls,
        train_env,
        policy_kwargs=policy_kwargs,
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
          f"(local={local_obs_dim}, global={global_dim}) "
          f"act_dim={train_env.action_space.shape} net=[{h1},{h2}] "
          f"policy={getattr(model.policy_class, '__name__', model.policy_class)} "
          f"steps={args.steps} device={args.device}")

    model.learn(total_timesteps=args.steps, callback=eval_callback)

    final_path = os.path.join(checkpoint_dir, "ppo_model.zip")
    model.save(final_path)
    print(f"[ppo] final model saved: {final_path}")
    print(f"[ppo] 训练完成。详细评估请运行: "
          f"python src/eval_ppo.py --checkpoint {final_path}")


if __name__ == "__main__":
    main()

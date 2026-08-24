"""TD3 训练主脚本（纯全连接，支持 action chunking）。

轮次制训练（对齐 compare/train_c.py）：epoch × episode。
    - 每 train_every_n 个 episode 训练一次（每次 training_iterations 次梯度更新）
    - 每个 epoch 完成 episodes_per_epoch 个 episode，之后评估一次
    - 每 save_every_epoch 个 epoch 保存 checkpoint；评估成功率更高时保存 model_best

N（chunk size）由 config 或 --chunk 指定；N=1 即 E1，N>1 即 E3。

并行环境（--parallel-workers N，N>1 时启用）：
    把 episode 采样分散到 N 个 worker 进程（每个进程一个 ir-sim 环境），
    主进程负责梯度更新与评估。瓶颈从单进程仿真转移到主进程消费，
    吞吐量近似线性提升（例如 8 worker 约 6-8 倍）。

用法:
    python train/train.py --config configs/train.yaml --name mlptd3 \
        --chunk 1 [--device cpu] [--seed 0] [--steps 上限(可选)] [--parallel-workers 8]
"""

from __future__ import annotations

import argparse
import os
import sys
from queue import Empty

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alg.buffer import ReplayBuffer  # noqa: E402
from alg.td3 import TD3Agent  # noqa: E402
from env.wrapper import IrSimEnv  # noqa: E402
from model.td3 import build_actor_critic  # noqa: E402
from utils.checkpoint import save_checkpoint  # noqa: E402
from utils.config import deep_update, load_config, resolve_path  # noqa: E402
from utils.logger import Logger  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_eval_policy(agent: TD3Agent, device: str | torch.device):
    def policy(obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
        return agent.select_action(obs_t, noise=0.0).numpy().reshape(
            agent.chunk_size, agent.action_dim
        )

    return policy


def _evaluate(
    env: IrSimEnv, agent: TD3Agent, chunk: int, device: str, seed: int, episodes: int
) -> dict:
    """在共享的评估环境上跑 episodes。

    评估环境由 :func:`train` 创建一次并复用（可视化），本函数不负责创建/关闭，
    评估结束后保留窗口供观察。
    """
    from eval.evaluate import evaluate_policy

    policy = make_eval_policy(agent, device)
    agent.actor.eval()
    try:
        metrics = evaluate_policy(
            env, policy, chunk_size=chunk, episodes=episodes, device=device, seed=seed + 1000
        )
    finally:
        agent.actor.train()
    return metrics


def _save_checkpoint(p: dict, global_step: int, epoch: int | None = None,
                     best: bool = False, best_success: float | None = None) -> None:
    path = os.path.join(p["checkpoint_dir"], "model_best.pt" if best else "model.pt")
    extra = {"best_success": best_success} if best else None
    save_checkpoint(
        path,
        p["cfg"],
        p["agent"].actor,
        p["agent"].critic1,
        p["agent"].critic2,
        step=global_step,
        tag=p["tag"],
        extra=extra,
    )


def _log_eval(p: dict, epoch: int, global_step: int, best_success: float) -> float:
    """评估一次并写日志/保存；返回更新后的 best_success。"""
    eval_metrics = _evaluate(
        p["eval_env"], p["agent"], p["chunk"], p["device"], p["seed"], p["eval_episodes"]
    )
    p["logger"].log_metrics(eval_metrics, global_step, prefix="eval")
    p["logger"].write_csv_row({"epoch": epoch, "step": global_step, **eval_metrics})
    success = float(eval_metrics.get("success_rate", 0.0))
    print(f"[eval] epoch={epoch} step={global_step} success={success:.3f}")

    if epoch % p["save_every_epoch"] == 0:
        _save_checkpoint(p, global_step, epoch=epoch)
    if success > best_success:
        best_success = success
        _save_checkpoint(p, global_step, epoch=epoch, best=True, best_success=best_success)
        print(f"[best] epoch={epoch} success={success:.3f}")
    return best_success


# --------------------------------------------------------------------------- #
# 串行训练（单环境，原逻辑）
# --------------------------------------------------------------------------- #
def _train_serial(p: dict) -> str:
    env = p["env"]
    buffer = p["buffer"]
    agent = p["agent"]
    logger = p["logger"]

    global_step = 0
    epoch, episode = 0, 0
    best_success = -1.0

    while epoch < p["max_epochs"] and (
        p["total_steps_cap"] is None or global_step < p["total_steps_cap"]
    ):
        # ---------------- 跑一个 episode ----------------
        obs = env.reset(random=True)
        ep_reward = 0.0
        ep_steps = 0
        done = False
        while not done and ep_steps < p["episode_max_steps"]:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=p["device"])
            action = agent.select_action(obs_t, noise=p["exploration_noise"]).numpy()
            next_obs, r_sum, done, info = env.step_chunk(action, gamma=p["gamma"])

            buffer.push(obs, action.reshape(-1), r_sum, next_obs, done)
            obs = next_obs
            ep_reward += r_sum
            ep_steps += 1
            global_step += 1

        episode += 1
        logger.log_scalar("train/episode_reward", ep_reward, global_step)
        logger.log_scalar("train/episode_steps", ep_steps, global_step)

        # ---------------- 每 train_every_n 个 episode 训练一次 ----------------
        if episode % p["train_every_n"] == 0 and len(buffer) >= p["batch_size"]:
            for _ in range(p["training_iterations"]):
                losses = agent.update(buffer.sample(p["batch_size"]))
            logger.log_metrics(losses, global_step, prefix="loss")

        # ---------------- 完成一个 epoch：评估 + 保存 ----------------
        if episode >= p["episodes_per_epoch"]:
            episode = 0
            epoch += 1
            best_success = _log_eval(p, epoch, global_step, best_success)

    # ---------------- 收尾保存 ----------------
    _save_checkpoint(p, global_step)
    if p["save_buffer"]:
        buffer.save(os.path.join(p["buffer_dir"], "buffer.npz"))

    logger.close()
    env.close()
    return p["tag"]


# --------------------------------------------------------------------------- #
# 并行训练（多进程采样，主进程更新/评估）
# --------------------------------------------------------------------------- #
def _train_parallel(p: dict) -> str:
    import multiprocessing as mp

    from train.parallel import worker_main

    num_workers = int(p["num_workers"])
    ctx = {
        "cfg": p["cfg"],
        "chunk": p["chunk"],
        "seed": p["seed"],
        "exploration_noise": p["exploration_noise"],
        "gamma": p["gamma"],
        "episode_max_steps": p["episode_max_steps"],
        "init_state_dict": {
            k: v.detach().cpu() for k, v in p["agent"].actor.state_dict().items()
        },
    }
    spawn = mp.get_context("spawn")
    result_queue = spawn.Queue(maxsize=10000)
    weight_queues = [spawn.Queue(maxsize=2048) for _ in range(num_workers)]
    procs = [
        spawn.Process(target=worker_main, args=(i, ctx, weight_queues[i], result_queue))
        for i in range(num_workers)
    ]
    for pr in procs:
        pr.daemon = True
        pr.start()

    version = 0

    def broadcast() -> None:
        nonlocal version
        version += 1
        sd = {k: v.detach().cpu().clone() for k, v in p["agent"].actor.state_dict().items()}
        msg = {"version": version, "state_dict": sd}
        for q in weight_queues:
            # 超时防止 worker 阻塞在 result_queue 时把主进程卡死；
            # 下一轮广播会覆盖，TD3 可容忍少量陈旧权重。
            try:
                q.put(msg, timeout=2.0)
            except Exception:
                pass

    broadcast()

    buffer = p["buffer"]
    agent = p["agent"]
    logger = p["logger"]

    global_step = 0
    epoch, episode = 0, 0
    best_success = -1.0

    def push_and_log(transitions, ep_reward, ep_steps) -> None:
        nonlocal global_step, episode
        for t in transitions:
            buffer.push(*t)
        global_step += len(transitions)
        episode += 1
        logger.log_scalar("train/episode_reward", ep_reward, global_step)
        logger.log_scalar("train/episode_steps", ep_steps, global_step)

    def train_if_needed() -> None:
        if episode % p["train_every_n"] == 0 and len(buffer) >= p["batch_size"]:
            for _ in range(p["training_iterations"]):
                losses = agent.update(buffer.sample(p["batch_size"]))
            logger.log_metrics(losses, global_step, prefix="loss")
            broadcast()

    def epoch_rollover() -> None:
        nonlocal epoch, episode, best_success
        while episode >= p["episodes_per_epoch"]:
            episode -= p["episodes_per_epoch"]
            epoch += 1
            best_success = _log_eval(p, epoch, global_step, best_success)

    def keep_going() -> bool:
        return epoch < p["max_epochs"] and (
            p["total_steps_cap"] is None or global_step < p["total_steps_cap"]
        )

    try:
        while keep_going():
            # 阻塞取一个 episode（带超时，避免 worker 偶发空闲时错过终止检查）
            try:
                worker_id, transitions, ep_reward, ep_steps = result_queue.get(timeout=1.0)
            except Empty:
                continue
            # 处理该 episode，再尽量取走已积压的；每个 episode 处理完都重查终止条件，
            # 防止 worker 生产比消费快时内层排空循环永远停不下来
            while True:
                push_and_log(transitions, ep_reward, ep_steps)
                train_if_needed()
                epoch_rollover()
                if not keep_going():
                    break
                try:
                    worker_id, transitions, ep_reward, ep_steps = result_queue.get_nowait()
                except Empty:
                    break
    finally:
        for q in weight_queues:
            try:
                q.put(None, timeout=2.0)
            except Exception:
                pass
        for pr in procs:
            pr.join(timeout=10)
        for pr in procs:
            if pr.is_alive():
                pr.terminate()

    # ---------------- 收尾保存 ----------------
    _save_checkpoint(p, global_step)
    if p["save_buffer"]:
        buffer.save(os.path.join(p["buffer_dir"], "buffer.npz"))

    logger.close()
    p["eval_env"].close()
    return p["tag"]


# --------------------------------------------------------------------------- #
def train(
    cfg: dict,
    name: str,
    chunk: int,
    steps: int | None,
    seed: int,
    device: str,
    num_workers: int | None = None,
) -> str:
    """轮次制训练（epoch × episode，对齐 compare/train_c.py）。

    Args:
        cfg: 配置。
        name: 实验名。
        chunk: action chunk 长度 N。
        steps: 全局决策步上限（可选；None 时完全由 train.max_epochs 控制）。
        seed: 随机种子。
        device: 设备。
        num_workers: 并行采样 worker 数；None 时取 config train.parallel_workers。
    """
    set_seed(seed)

    train_cfg = cfg["train"]
    td3_cfg = cfg["td3"]
    chunk_gamma = td3_cfg["gamma"]
    exploration_noise = td3_cfg["exploration_noise"]
    batch_size = td3_cfg["batch_size"]

    max_epochs = int(train_cfg.get("max_epochs", 120))
    episodes_per_epoch = int(train_cfg.get("episodes_per_epoch", 70))
    train_every_n = int(train_cfg.get("train_every_n", 2))
    training_iterations = int(train_cfg.get("training_iterations", 80))
    episode_max_steps = int(train_cfg.get("max_steps", 300))
    eval_episodes = int(train_cfg.get("eval_episodes", 10))
    save_every_epoch = int(train_cfg.get("save_every_epoch", 10))
    total_steps_cap = int(steps) if steps is not None else None
    if num_workers is None:
        num_workers = int(train_cfg.get("parallel_workers", 1))

    # ---------------- 环境 ----------------
    if cfg["obs"].get("include_prev_action"):
        cfg = deep_update(cfg, {"obs": {"prev_chunk_size": chunk}})
    env = IrSimEnv(
        resolve_path(cfg["env"]["world_name"]),
        reward_cfg=cfg["reward"],
        obs_cfg=cfg["obs"],
        env_cfg=cfg["env"],
        seed=seed,
        display=cfg["env"].get("display", False),
        log_level="CRITICAL",
    )
    # 轮次制以「决策步」计 episode 长度；放宽环境控制步超时，避免 chunk 时提前截断
    env.max_steps = max(int(env.max_steps), episode_max_steps * chunk)

    cfg = deep_update(cfg, {"obs": {"lidar_len": env.lidar_len}})
    goal_dim = env.goal_dim if env.include_goal else 0
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    if cfg["obs"].get("include_prev_action"):
        cfg = deep_update(cfg, {"model": {"prev_dim": chunk * action_dim}})

    # 评估环境：整个训练只创建一次并复用。
    eval_env = IrSimEnv(
        resolve_path(
            cfg["env"].get("eval_world", cfg["env"].get("world_name_evaluate", cfg["env"]["world_name"]))
        ),
        reward_cfg=cfg["reward"],
        obs_cfg=cfg["obs"],
        env_cfg=cfg["env"],
        seed=seed + 1000,
        display=cfg["env"].get("display", False),
        log_level="CRITICAL",
    )

    # ---------------- 模型 / 智能体 ----------------
    actor, critic1, critic2 = build_actor_critic(
        lidar_len=env.lidar_len,
        goal_dim=goal_dim,
        action_dim=action_dim,
        chunk_size=chunk,
        model_cfg=cfg["model"],
        quant_ready=True,
    )
    agent = TD3Agent(actor, critic1, critic2, cfg["td3"], chunk_size=chunk, device=device)

    buffer = ReplayBuffer(
        cfg["td3"]["buffer_size"], obs_dim, action_dim * chunk, seed=seed
    )

    tag = f"{name}_N{chunk}"
    checkpoint_dir = resolve_path(os.path.join(cfg["project"]["checkpoint_dir"], tag))
    buffer_dir = resolve_path(os.path.join(cfg["project"]["buffer_dir"], tag))
    logger = Logger(resolve_path(os.path.join(cfg["project"]["run_dir"], tag)))

    print(f"[train] tag={tag} obs_dim={obs_dim} action_dim={action_dim} chunk={chunk}")
    print(
        f"[train] 轮次制: epochs={max_epochs} episodes/epoch={episodes_per_epoch} "
        f"train_every_n={train_every_n} iterations={training_iterations}"
    )
    print(f"[train] parallel_workers={num_workers}")

    p = dict(
        cfg=cfg,
        agent=agent,
        buffer=buffer,
        logger=logger,
        tag=tag,
        checkpoint_dir=checkpoint_dir,
        buffer_dir=buffer_dir,
        eval_env=eval_env,
        chunk=chunk,
        device=device,
        seed=seed,
        gamma=chunk_gamma,
        exploration_noise=exploration_noise,
        batch_size=batch_size,
        max_epochs=max_epochs,
        episodes_per_epoch=episodes_per_epoch,
        train_every_n=train_every_n,
        training_iterations=training_iterations,
        episode_max_steps=episode_max_steps,
        eval_episodes=eval_episodes,
        save_every_epoch=save_every_epoch,
        total_steps_cap=total_steps_cap,
        save_buffer=train_cfg.get("save_buffer", True),
        num_workers=num_workers,
    )

    if num_workers > 1:
        env.close()  # probe env：并行时由 worker 各自持有环境
        p["env"] = None
        return _train_parallel(p)

    p["env"] = env
    return _train_serial(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="TD3（纯全连接 MLP）训练")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--name", default="mlptd3")
    ap.add_argument("--chunk", type=int, default=None, help="覆盖 config 的 chunk.size")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--parallel-workers", type=int, default=None,
                    help="并行采样 worker 数；>1 启用多进程并行环境（默认取 config train.parallel_workers）")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    if args.chunk is not None:
        cfg = deep_update(cfg, {"chunk": {"size": args.chunk}})
    chunk = int(cfg["chunk"]["size"])
    steps = args.steps  # 可选全局步数上限；None 时由 train.max_epochs 控制

    tag = train(cfg, args.name, chunk, steps, args.seed, args.device,
                num_workers=args.parallel_workers)
    print(f"[done] trained tag={tag}")


if __name__ == "__main__":
    main()

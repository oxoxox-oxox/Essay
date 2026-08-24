"""打印 TD3 模型的规模（参数量 / MACs / 文件大小）。用法: python eval/_model_size.py [tag1 tag2 ...]，默认 mlptd3_N5 mlptd3_1024_N5"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.checkpoint import load_checkpoint  # noqa: E402
from eval.policy import build_actor  # noqa: E402


def count_params(m):
    return sum(p.numel() for p in m.parameters())


TAGS = sys.argv[1:] or ["mlptd3_N5", "mlptd3_1024_N5"]

for tag in TAGS:
    ckpt = f"checkpoints/{tag}/model_best.pt"
    if not os.path.exists(ckpt):
        print(f"== {tag} ==  (checkpoint 不存在: {ckpt})")
        continue
    data = load_checkpoint(ckpt, device="cpu")
    meta = data["meta"]
    mc = meta.get("model_cfg", {})
    actor = build_actor(meta, mc)
    obs = meta["lidar_len"] + meta["goal_dim"] + int(mc.get("prev_dim", 0))
    act_total = meta["action_dim"] * meta["chunk_size"]
    cin = obs + act_total  # critic 输入 = obs + 展平动作
    h1, h2 = int(mc.get("hidden1", 400)), int(mc.get("hidden2", 300))

    a_params = count_params(actor)
    c_params = (cin * h1 + h1) + (h1 * h2 + h2) + (h2 * 1 + 1)
    a_macs = obs * h1 + h1 * h2 + h2 * act_total
    c_macs = cin * h1 + h1 * h2 + h2 * 1
    tot_online = a_params + 2 * c_params

    print(f"== {tag} ==")
    print(f"  obs_dim={obs}  action=(N={meta['chunk_size']}, dim={meta['action_dim']})  actor输出={act_total}")
    print(f"  层宽: {h1} -> {h2}")
    print(f"  Actor 参数量    : {a_params:,}  ({a_params/1e6:.3f}M)")
    print(f"  单个 Critic 参数: {c_params:,}  ({c_params/1e6:.3f}M)  x2 = {2*c_params:,}")
    print(f"  在线网络合计     : {tot_online:,}  ({tot_online/1e6:.3f}M)")
    print(f"  (含 target 全量 : {2*tot_online:,}  ({2*tot_online/1e6:.3f}M))")
    print(f"  Actor MACs/推理 : {a_macs:,}  ({a_macs/1e6:.3f}M)")
    print(f"  单 Critic MACs  : {c_macs:,}   双 critic 合计: {2*c_macs:,}")
    print(f"  checkpoint 文件 : {os.path.getsize(ckpt)/1e6:.2f} MB")
    print()

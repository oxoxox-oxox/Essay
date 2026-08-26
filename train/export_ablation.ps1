# 一键导出消融实验的两个模型（N=1 与 N=5）为 ONNX + INT8 校准数据。
# 前置：checkpoints/ppo_83ms_N1/best_model.zip 与 checkpoints/ppo_83ms_N5/best_model.zip 已训练完成。
# 用法（PowerShell，仓库根目录）:  .\train\export_ablation.ps1

$rl = "d:\anaconda\envs\rl_env\python.exe"
$irs = "d:\anaconda\envs\ir-sim\python.exe"

Write-Host "== N=1 (obs 105, act 2) =="
& $rl train\unpack_ppo_actor.py --checkpoint checkpoints/ppo_83ms_N1/best_model.zip --name ppo_83ms_N1
& $irs train\export_ppo_onnx.py --actor export/ppo_83ms_N1/policy_actor.pt --make-calib

Write-Host "== N=5 (obs 113, act 10) =="
& $rl train\unpack_ppo_actor.py --checkpoint checkpoints/ppo_83ms_N5/best_model.zip --name ppo_83ms_N5
& $irs train\export_ppo_onnx.py --actor export/ppo_83ms_N5/policy_actor.pt --make-calib --chunk 5

Write-Host "== 完成。产物在 export/ppo_83ms_N1/ 与 export/ppo_83ms_N5/ =="
Write-Host "== 记得把 export/ppo_mw/build_ptq_engine.py 复制进这两个目录（板上建 INT8 引擎用）=="

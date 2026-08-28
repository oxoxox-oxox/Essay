# One-shot export of both ablation experiment models (N=1 and N=5) to ONNX + INT8 calibration data.
# Prerequisite: checkpoints/ppo_final_N1/best_model.zip and checkpoints/ppo_final_N5/best_model.zip are already trained.
# Usage (PowerShell, repo root):  .\train\export_ablation.ps1

$rl = "d:\anaconda\envs\rl_env\python.exe"
$irs = "d:\anaconda\envs\ir-sim\python.exe"

Write-Host "== N=1 (obs 105, act 2) =="
& $rl train\unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N1/best_model.zip --name ppo_final_N1
& $irs train\export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib

Write-Host "== N=5 (obs 113, act 10) =="
& $rl train\unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N5/best_model.zip --name ppo_final_N5
& $irs train\export_ppo_onnx.py --actor export/ppo_final_N5/policy_actor.pt --make-calib --chunk 5

Write-Host "== Done. Artifacts in export/ppo_final_N1/ and export/ppo_final_N5/ =="
Write-Host "== Build on-board INT8 engines with deploy/ppo_nav/scripts/build_ptq_engine.py =="

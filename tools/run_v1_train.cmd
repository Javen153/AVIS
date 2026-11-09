@echo off
rem Detached v1 training launcher (used via tools\bg.ps1 -Name v1).
rem Resumes from models\v1\ckpt_last.pt after the step-4220 divergence (DECISIONS D-015).
cd /d "E:\OpenCode\AVIS 1.0\ml"
set PYTHONPATH=E:\OpenCode\AVIS 1.0\ml
".venv\Scripts\python.exe" -u -m avis_ml.train --config configs/v1.yaml --resume

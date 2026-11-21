@echo off
REM Full sweep on a Windows workstation: data prep, then one CUDA worker + one CPU worker
REM (+ optional DirectML worker for integrated GPUs). Safe to re-run: it resumes.
set TOPO_ROOT=%~dp0..\work
set PY=%~dp0..\venv\Scripts\python.exe
cd /d %~dp0
%PY% prep_data.py
%PY% orchestrate.py --gpu_workers 1 --cpu_workers 1 --cpu_threads 2 --tag laptop
%PY% agg.py laptop

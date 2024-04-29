@echo off
call "C:\Users\intel\miniconda3\Scripts\activate.bat" C:\Users\intel\miniconda3\envs\openvino_env_py310
python src/run_rag.py
pause
call "C:\Users\intel\miniconda3\Scripts\deactivate.bat"

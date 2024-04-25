@echo off
call "C:\Users\intel\miniconda3\Scripts\activate.bat" C:\Users\intel\miniconda3\envs\openvino_env_py310
set HTTP_PROXY=http://proxy-us.intel.com:914
set HTTPS_PROXY=http://proxy-us.intel.com:914
python run_rag.py
pause
call "C:\Users\intel\miniconda3\Scripts\deactivate.bat"

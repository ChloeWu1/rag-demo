@echo off

echo Setting up proxy...
set HTTP_PROXY=http://proxy-us.intel.com:914
set HTTPS_PROXY=http://proxy-us.intel.com:914

echo Creating Python environment...
call "C:\Users\intel\miniconda3\Scripts\conda" create -n test2 python=3.10 -y

echo Activating environment...
call "C:\Users\intel\miniconda3\Scripts\activate.bat" test2

echo Installing requirements...
pip install --extra-index-url https://download.pytorch.org/whl/cpu ^
git+https://github.com/huggingface/optimum-intel.git ^
git+https://github.com/openvinotoolkit/nncf.git ^
datasets ^
accelerate ^
openvino-nightly ^
gradio ^
onnx ^
einops ^
transformers_stream_generator ^ 
tiktoken ^
transformers>=4.38.1 ^
bitsandbytes ^
chromadb ^
sentence_transformers ^
langchain>=0.1.15 ^
langchainhub ^
unstructured ^
scikit-learn ^
python-docx ^
pdfminer.six ^
python-pptx ^
markdown ^
pypdf

echo Checking installed packages...
conda list

echo Installation complete!

cmd /k
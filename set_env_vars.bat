@echo off
rem 克隆 openvino_notebooks 仓库
git clone openvino_notebooks
rem 进入 openvino_notebooks 文件夹
cd openvino_notebooks
rem 安装 requirements.txt 中的依赖
pip install -r requirements.txt
rem 返回到工作文件夹
cd ..
rem 安装 rag_requirements.txt 中的依赖
pip install -r rag-requirements.txt

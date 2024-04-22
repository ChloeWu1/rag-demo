# Create a RAG system using OpenVINO and LangChain

## System 

Windows 11

## Installation

1. Download Miniconda

   - [Miniconda Installation Guide](https://docs.anaconda.com/free/miniconda/)
   - Download Link: [Miniconda3-latest-Windows-x86_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

2. Install Miniconda
   
   - Install for "Just Me"
   - Verify the installation path: C:\Users\"username"\miniconda3
   - Open Anaconda Prompt

3. Configure Conda Environment
   ```
   conda deactivate  # exit the base environment
   conda create -n openvino_env python=3.10
   conda activate openvino_env
   ```

4. Setup Working Environment
   ```
   mkdir openvino_prj 
   在桌面新建文件夹，命名为openvino_prj. 将rag-requirements.txt, run_rag.py, set_env_vars.bat放进文件夹里。
   
   在终端中：
   cd openvino_prj 
   set_env_vars.bat
   ```

   ```
   set PATH=C:\Users\"username"\Miniconda3\envs\openvino_env\Lib\site-packages\openvino\libs;%PATH% # 每次新开promt都要跑一下，如果确定用户名默认是intel，可以写进bat里。如果每台机器用户不一样，手动设置。
   
   set HTTP_PROXY=proxy:port #不下载模型不需要
   set HTTPS_PROXY=proxy:port #不下载模型不需要
   ```

5. Running the Application
   ```
   python run_rag.py
   ```


**Appendix: Package Versions**
This appendix lists the versions of all installed packages for your reference.

| Package                             | Version          |
| ----------------------------------- | -----------------|
| accelerate                          | 0.29.2           |
| bitsandbytes                        | 0.43.1           |
| chromadb                            | 0.4.24           |
| datasets                            | 2.18.0           |
| einops                              | 0.7.0            |
| gradio                              | 3.50.0           |
| gradio_client                       | 0.6.1            |
| huggingface-hub                     | 0.22.2           |
| langchain                           | 0.1.16           |
| langchainhub                        | 0.1.15           |
| nncf                                | 2.10.0.dev0+573b0c34 |
| onnx                                | 1.16.0           |
| openvino-nightly                    | 2024.2.0.dev20240416  |
| optimum                             | 1.19.0           |
| optimum-intel                       | 1.17.0.dev0+ff5d185  |
| pdfminer.six                        | 20231228         |
| python-docx                         | 1.1.0            |
| scikit-learn                        | 1.4.2            |
| sentence-transformers               | 2.6.1            |
| tiktoken                            | 0.6.0            |
| transformers                        | 4.39.3           |
| transformers-stream-generator       | 0.0.5            |
| unstructured                        | 0.13.2           |
| unstructured-client                 | 0.18.0           |
| urllib3                             | 2.2.1            |
| wheel                               | 0.41.2           |  

This README provides instructions for setting up Miniconda, configuring the Conda environment, and running an application on a Windows 11 system. Follow the steps above to ensure proper setup and execution of the application. 
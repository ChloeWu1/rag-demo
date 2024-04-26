# Create a RAG system using OpenVINO and LangChain

## System 

Windows 11

## Installation

1. Download Miniconda

   - [Miniconda Installation Guide](https://docs.anaconda.com/free/miniconda/)
   - Download Link: [Miniconda3-latest-Windows-x86_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

2. Install Miniconda
   
   - Install Miniconda for the `Just Me` option
   - Verify the installation path in the environment variable
      ```
      C:\Users\"your_username"\miniconda3
      ```
   - Open Anaconda Prompt

3. Configure Conda Environment
   ```
   conda deactivate  # exit the base environment
   conda create -n openvino_env_py310 python=3.10
   conda activate openvino_env_py310
   ```

4. Setup Working Environment
   Create a new folder on the desktop, named `openvino-prj`  
   Run under `openvino_env_py310` environment in Anaconda Prompt:  
   ```
   cd Desktop/openvino-prj
   pip install -r rag-requirements.txt
   ```

5. Setup Openvino-pkg path and Proxy  

   - Set openvino-pkg path
   ```
   set PATH=C:\Users\"your_username"\Miniconda3\envs\openvino_env_py310\Lib\site-packages\openvino\libs;%PATH%
   ```
   - Set Proxy (Optional)  
   If you want to download or convert the model, please ensure that you can access huggingface.
   ```
   set HTTP_PROXY=http://proxy-us.intel.com:914
   set HTTPS_PROXY=http://proxy-us.intel.com:914
   ```

## Running the Application
   Double click:
   ```
   run_rag.bat
   ```

## Download and convert models
   Run script in the env:
   ```
   python download_convert_model.py
   ```
   Select the Chinese model by default, `--language` parameter can specify the model language.

   ```
   python download_convert_model.py --language=English
   ```

## How to select models
   Please check `utils/llm_config.py` and `run_rag.py`.  
   Modify Line59, Line64, Line87, Line101, Line 159 in the `run_rag.py`  

## Appendix

### Package Versions
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

### Project Structure
```text
openvino-prj/
│
├── download_convert_model.py
├── rag-requirements.txt
├── README.md
├── run_rag.bat
├── run_rag.py
│
├── bge-reranker-large/
│
├── bge-small-en-v1.5/
│
├── bge-small-zh-v1.5/
│
├── model/
│   ├── cache_dir/
│   └── qwen1.5-7b-chat/
│
└── utils/
    ├── llm_config.py
    ├── logo_trans.png
    └── __init__.py
```

This README provides instructions for setting up Miniconda, configuring the Conda environment, and running an application on a Windows 11 system. Follow the steps above to ensure proper setup and execution of the application. 

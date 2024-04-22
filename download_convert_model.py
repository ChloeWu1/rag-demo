import argparse
import subprocess
from pathlib import Path
import os
from transformers import AutoTokenizer
from llm_config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_RERANK_MODELS,
    SUPPORTED_LLM_MODELS,
)

# 定义INT4的压缩配置
compression_configs = {
        "zephyr-7b-beta": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "mistral-7b": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "minicpm-2b-dpo": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "gemma-2b-it": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "notus-7b-v1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "neural-chat-7b-v3-1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "llama-2-chat-7b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "llama-3-8b-instruct": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "gemma-7b-it": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "chatglm2-6b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.72,
        },
        "qwen-7b-chat": {"sym": True, "group_size": 128, "ratio": 0.6},
        "red-pajama-3b-chat": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.5,
        },
        "default": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.8,
        },
    }

# 命令行参数解析
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--language', choices=['English', 'Chinese', 'Japanese'], default='Chinese',
                    help='Language of the model (default: English)')

args = parser.parse_args()

model_language = args.language
llm_model_ids = [model_id for model_id, model_config in SUPPORTED_LLM_MODELS[model_language].items() if model_config.get("rag_prompt_template")]
print("Available models are:", llm_model_ids)

tokenizer = None
while tokenizer is None:
    input_model_name = input("请输入模型名称：")
    model_name = None

    for model_id, model_config in SUPPORTED_LLM_MODELS[model_language].items():
        if input_model_name == model_id:
            model_name = model_config["model_id"]
            break

    if model_name is None:
        print("输入的模型名称无效，请重新输入。")
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"下载或加载模型时出现错误：{e}")

    weight_format = input("请输入权重格式（fp16, int8, int4）：")
    if weight_format not in ["fp16", "int8", "int4"]:
        print("权重格式错误，请输入有效的格式。")
        continue

    compression_config = compression_configs.get(model_name, compression_configs["default"])

    try:
        # 生成输出路径
        fp16_model_dir = Path(input_model_name) / "FP16"
        int8_model_dir = Path(input_model_name) / "INT8_compressed_weights"
        int4_model_dir = Path(input_model_name) / "INT4_compressed_weights"

        if weight_format == "fp16":
            output_path = fp16_model_dir
        elif weight_format == "int8":
            output_path = int8_model_dir
        elif weight_format == "int4":
            output_path = int4_model_dir

        cmd = f"optimum-cli export openvino --model {model_name} --task text-generation-with-past --weight-format {weight_format} {output_path}"

        if weight_format == "int4":
            cmd += f" --group-size {compression_config['group_size']} --ratio {compression_config['ratio']}"
            if compression_config.get("sym"):
                cmd += " --sym"

        subprocess.run(cmd, shell=True, check=True)
        
        print("模型转换成功！")
    except Exception as e:
        print(f"转换模型时出现错误：{e}")

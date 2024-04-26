import argparse
import subprocess
from pathlib import Path
from transformers import AutoTokenizer
from utils.llm_config import SUPPORTED_LLM_MODELS

# Define compression configs for INT4
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

# Add arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--language', choices=['English', 'Chinese', 'Japanese'], default='Chinese',
                    help='Language of the model (default: English)')

args = parser.parse_args()

model_language = args.language
llm_model_ids = [model_id for model_id, model_config in SUPPORTED_LLM_MODELS[model_language].items() if model_config.get("rag_prompt_template")]
print("列出可用的模型", llm_model_ids)

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
            print(f"下载或加载模型时出现错误:{e}")

    weight_format = input("请输入权重格式(fp16, int8, int4):")
    if weight_format not in ["fp16", "int8", "int4"]:
        print("权重格式错误，请输入有效的格式。")
        continue

    compression_config = compression_configs.get(model_name, compression_configs["default"])

    try:
        # Generate output_path
        model_dir = Path("model") / input_model_name
        model_dir.mkdir(parents=True, exist_ok=True)  # 确保目录创建，包括其所有父目录

        fp16_model_dir = model_dir / "FP16"
        int8_model_dir = model_dir / "INT8_compressed_weights"
        int4_model_dir = model_dir / "INT4_compressed_weights"

        fp16_model_dir.mkdir(exist_ok=True)
        int8_model_dir.mkdir(exist_ok=True)
        int4_model_dir.mkdir(exist_ok=True)

        if weight_format == "fp16":
            output_path = fp16_model_dir
        elif weight_format == "int8":
            output_path = int8_model_dir
        elif weight_format == "int4":
            output_path = int4_model_dir
        print("模型的输出路径是：", output_path)

        cmd = f"optimum-cli export openvino --model {model_name} --task text-generation-with-past --weight-format {weight_format} {output_path}"

        if weight_format == "int4":
            cmd += f" --group-size {compression_config['group_size']} --ratio {compression_config['ratio']}"
            if compression_config.get("sym"):
                cmd += " --sym"

        subprocess.run(cmd, shell=True, check=True)

        print("模型转换成功！")
    except Exception as e:
        print(f"转换模型时出现错误：{e}")

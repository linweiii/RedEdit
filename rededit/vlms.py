import base64
import requests
import json
from pathlib import Path
import os, sys
import pandas as pd
import torch
from PIL import Image
import requests
from io import BytesIO

# SiliconFlow API key — set via environment variable SILICONFLOW_API_KEY
# Get your key at: https://cloud.siliconflow.cn
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# OpenAI-compatible API key (GPT-4V)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── VLM registry ─────────────────────────────────────────────────────────
# Model name strings as required by SiliconFlow API.
# Available Qwen3-VL sizes on SiliconFlow: 8B, 32B, 235B-A22B
# See full model list: https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions
VLM2ModelPaths = {
    "gpt-4v":        "gpt-4-vision-preview",
    "instructblip-7b": "",
    "qwen3-vl-8b":   "Qwen/Qwen3-VL-8B",    # ~9B
    "qwen3-vl-32b":  "Qwen/Qwen3-VL-32B",   # ~27B / ~122B
    "qwen3-vl-235b": "Qwen/Qwen3-VL-235B-A22B",  # ~235B / ~397B
}

# All Qwen models are served via the unified SiliconFlow endpoint.
# GPT-4V uses the standard OpenAI endpoint.
VLM2URL = {
    "gpt-4v":        "https://api.openai.com/v1",
    "qwen3-vl-8b":   SILICONFLOW_BASE_URL,
    "qwen3-vl-32b":  SILICONFLOW_BASE_URL,
    "qwen3-vl-235b": SILICONFLOW_BASE_URL,
}


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_path):
    if image_path.startswith("http") or image_path.startswith("https"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image

def load_images(image_paths):
    out = []
    for image_path in image_paths:
        image = load_image(image_path)
        out.append(image)
    return out

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_type(image_path):
    """ Returns the type of the image (e.g., JPEG, PNG) """
    with Image.open(image_path) as img:
        return img.format.lower()

def load_vlm(model_name):

    if "gpt" in model_name:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client

    elif "qwen" in model_name:
        from openai import OpenAI
        api_key = SILICONFLOW_API_KEY
        if not api_key:
            raise ValueError(
                "SILICONFLOW_API_KEY is not set. "
                "Get your key at https://cloud.siliconflow.cn and run: "
                "export SILICONFLOW_API_KEY=your_key"
            )
        if model_name not in VLM2URL:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(VLM2URL.keys())}"
            )
        client = OpenAI(api_key=api_key, base_url=VLM2URL[model_name])
        return client

    elif "llava" in model_name:
        # LLaVA requires the separate 'llava' conda environment and local model weights.
        # This pack does NOT include LLaVA weights. Use Qwen models instead.
        raise NotImplementedError(
            "LLaVA is not supported in this pack. "
            "Use a Qwen model (e.g. --model_name qwen3-vl-8b) instead. "
            "If you need LLaVA, set up the full UnsafeBench environment with the 'llava' conda env."
        )

    elif "instructblip" in model_name:
        # InstructBLIP requires the separate 'lavis' conda environment.
        # This pack does NOT include InstructBLIP weights.
        raise NotImplementedError(
            "InstructBLIP is not supported in this pack. "
            "Use a Qwen model (e.g. --model_name qwen3-vl-8b) instead. "
            "If you need InstructBLIP, set up the full UnsafeBench environment with the 'lavis' conda env."
        )

def inference(model_name, model, image_path, prompt, **gen_kwargs):

    if "gpt" in model_name or "qwen" in model_name:

        gen_kwargs = {
            "temperature": gen_kwargs["temperature"],
            "max_tokens":  gen_kwargs["max_new_tokens"],
            "top_p":       gen_kwargs["top_p"],
        }

        base64_image = encode_image(image_path)

        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]}]

        if "qwen3-vl" in model_name:
            # SiliconFlow passes enable_thinking at the top level, not inside chat_template_kwargs
            completion = model.chat.completions.create(
                model=VLM2ModelPaths[model_name],
                messages=messages,
                extra_body={"enable_thinking": False},
                **gen_kwargs
            )
        else:
            completion = model.chat.completions.create(
                model=VLM2ModelPaths[model_name],
                messages=messages,
                **gen_kwargs
            )
        output = completion.choices[0].message.content
        return output

    elif "llava" in model_name or "instructblip" in model_name:
        raise NotImplementedError(
            f"{model_name} is not supported in this pack. Use a Qwen model instead."
        )

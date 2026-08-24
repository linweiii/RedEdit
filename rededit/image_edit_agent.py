#!/usr/bin/env python3
"""
Image Editing Agent — Educational Reference Implementation
==========================================================
A self-contained image editing agent built on Qwen-Agent.

Architecture overview:
  Config  (API endpoints, model config)
    └─► Tool definitions  (BaseTool subclasses, registered via @register_tool)
          └─► ImageEditAgent  (wraps qwen_agent.Assistant, manages session state)
                └─► main()   (parse args → run one prompt → print result)

Usage:
  python image_edit_agent.py --prompt "generate a sunset photo"
  python image_edit_agent.py --image /path/to/photo.jpg --prompt "rotate 45 degrees"
  python image_edit_agent.py -i a.jpg -i b.jpg --prompt "blend the two images"
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import colorsys
import hashlib
import io
import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import json5
import numpy as np
import requests
from PIL import (Image, ImageDraw, ImageEnhance,
                 ImageFilter, ImageFont, ImageOps)

from qwen_agent.agents import Assistant
from qwen_agent.gui.utils import convert_fncall_to_text
from qwen_agent.llm.schema import ASSISTANT, CONTENT, FUNCTION, NAME, ROLE
from qwen_agent.tools.base import BaseTool, register_tool

# ---------------------------------------------------------------------------
# ① Configuration  ── edit these values to point at your model servers
# ---------------------------------------------------------------------------

# Language-only model — used when no images are attached to the turn.
LLM_CONFIG: Dict[str, Any] = {
    "model_type": "qwenvl_oai",
    "model": "Qwen/Qwen3-32B",
    "model_server": "https://api.siliconflow.cn/v1",
    "api_key":  os.getenv("SILICONFLOW_API_KEY", ""),
    "generate_cfg": {"top_p": 0.1, "temperature": 0, "seed": 42},
}

# Vision-language model — used when images are attached to the turn.
VLM_CONFIG: Dict[str, Any] = {
    "model_type": "qwenvl_oai",
    "model": "Qwen/Qwen3-VL-32B-Instruct",
    "model_server": "https://api.siliconflow.cn/v1",
    "api_key":  os.getenv("SILICONFLOW_API_KEY", ""),
    "generate_cfg": {
        "top_p": 0.1,
        "top_k": 20,
        "temperature": 0,
        "repetition_penalty": 1.0,
        "seed": 42,
    },
}

# Set True to attach image data to every message (requires a VLM endpoint).
# Set False to pass only file paths as text (works with a plain LLM).
USE_VLM = True

# ---------------------------------------------------------------------------
# ② System Prompt  ── describes role, tools, and path-chaining rules
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an intelligent image editing assistant. Understand the user's intent
and autonomously plan a sequence of tool calls to accomplish it.

## Capabilities
Pixel-level tools for non-destructive, representation-preserving edits:
rotate, flip, resize, thumbnail, brightness, contrast, saturation, hue,
grayscale, sepia, sharpness, vignette, compress, convert format, watermark, border.

## Workflow
1. Analyse the user's request.
2. Choose the minimal sequence of tools needed.
3. Call them one by one; pass the `output_path` from each tool as the
   `image_path` input of the next tool.
4. Summarise what was done and state the final image path.

## Path rules
- ALWAYS use the exact `output_path` returned by the previous tool.
  Never construct or guess file paths.
- Spatial tools use tool-specific parameter schemas; do not invent coordinates unless a tool explicitly asks for them.
""".strip()

# ---------------------------------------------------------------------------
# ③ Path helper
# ---------------------------------------------------------------------------

def _output_path(input_path: str, suffix: str) -> str:
    """Append *suffix* before the file extension to derive an output path.

    Example:
        _output_path("/tmp/cat.png", "_rotate90") -> "/tmp/cat_rotate90.png"

    Always returns an absolute path in the same directory as the input.
    """
    input_path = os.path.abspath(input_path)
    stem, ext = os.path.splitext(os.path.basename(input_path))
    return os.path.join(os.path.dirname(input_path), f"{stem}{suffix}{ext}")


# ---------------------------------------------------------------------------
# ④ Tool definitions
# ---------------------------------------------------------------------------
# Pattern for every tool:
#   @register_tool('name')          → registers the name in qwen-agent's registry
#   description                     → shown verbatim to the LLM as the tool's purpose
#   parameters                      → JSON-Schema-style list; the LLM fills these in
#   call(self, params: str, **kwargs)→ receives a JSON string, returns a JSON string
#                                      the return value must contain "output_path"
# ---------------------------------------------------------------------------

# ── Pure-Pillow editing tools ────────────────────────────────────────────────

@register_tool("image_rotate")
class ImageRotate(BaseTool):
    description = "Rotate an image clockwise by the given angle (degrees)."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "angle", "type": "integer", "description": "Clockwise rotation in degrees.", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        out = _output_path(p["image_path"], f"_rotate{p['angle']}")
        Image.open(p["image_path"]).rotate(-p["angle"], expand=True).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_flip")
class ImageFlip(BaseTool):
    description = 'Flip an image. direction: "horizontal" or "vertical".'
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "direction", "type": "string", "description": '"horizontal" or "vertical".', "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        img = Image.open(p["image_path"])
        ops = {"horizontal": Image.FLIP_LEFT_RIGHT, "vertical": Image.FLIP_TOP_BOTTOM}
        if p["direction"] not in ops:
            return json5.dumps({"status": "error", "error": "direction must be 'horizontal' or 'vertical'."}, ensure_ascii=False)
        out = _output_path(p["image_path"], f"_flip{p['direction']}")
        img.transpose(ops[p["direction"]]).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_resize")
class ImageResize(BaseTool):
    description = "Resize an image to the given width × height (pixels)."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "width",  "type": "integer", "description": "Target width.",  "required": True},
        {"name": "height", "type": "integer", "description": "Target height.", "required": True},
        {"name": "keep_aspect_ratio", "type": "boolean", "description": "Preserve aspect ratio (default false).", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        img = Image.open(p["image_path"])
        w, h = p["width"], p["height"]
        if p.get("keep_aspect_ratio", False):
            img.thumbnail((w, h), Image.Resampling.LANCZOS)
        else:
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        out = _output_path(p["image_path"], f"_resize{w}x{h}")
        img.save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_crop")
class ImageCrop(BaseTool):
    description = (
        "Crop a rectangular region. "
        "bbox_2d uses normalised 0–1000 coordinates: [x1, y1, x2, y2]."
    )
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "bbox_2d", "type": "array", "items": {"type": "number"},
         "description": "[x1, y1, x2, y2] in 0–1000 normalised coordinates.", "required": True},
        {"name": "label", "type": "string", "description": "Human-readable region label (for logging).", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        img = Image.open(p["image_path"])
        W, H = img.size
        x1, y1, x2, y2 = [int(c / 1000.0 * d) for c, d in zip(p["bbox_2d"], [W, H, W, H])]
        x1, x2 = sorted([max(0, min(x1, W)), max(0, min(x2, W))])
        y1, y2 = sorted([max(0, min(y1, H)), max(0, min(y2, H))])
        if x2 <= x1 or y2 <= y1:
            return json5.dumps({"status": "error", "error": "Degenerate bounding box."}, ensure_ascii=False)
        out = _output_path(p["image_path"], f"_crop{x1}_{y1}_{x2}_{y2}")
        img.crop((x1, y1, x2, y2)).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_mosaic")
class ImageMosaic(BaseTool):
    description = (
        "Pixelate (mosaic) a region of an image. "
        "bbox_2d uses normalised 0–1000 coordinates."
    )
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "bbox_2d", "type": "array", "items": {"type": "number"},
         "description": "[x1, y1, x2, y2] in 0–1000 normalised coordinates.", "required": True},
        {"name": "label", "type": "string", "description": "Region label (for logging).", "required": False},
        {"name": "mosaic_size", "type": "integer", "description": "Pixel block size (default 10).", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        img = Image.open(p["image_path"])
        W, H = img.size
        block = p.get("mosaic_size", 10)
        x1, y1, x2, y2 = [int(c / 1000.0 * d) for c, d in zip(p["bbox_2d"], [W, H, W, H])]
        x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
        y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return json5.dumps({"status": "error", "message": "mosaic region too small (zero area)"}, ensure_ascii=False)
        region = img.crop((x1, y1, x2, y2))
        small = region.resize((max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)), Image.NEAREST)
        pixelated = small.resize((x2 - x1, y2 - y1), Image.NEAREST)
        result = img.copy()
        result.paste(pixelated, (x1, y1))
        out = _output_path(p["image_path"], "_mosaic")
        result.save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_adjust_brightness")
class ImageAdjustBrightness(BaseTool):
    description = "Adjust brightness. factor > 1 = brighter, < 1 = darker."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "factor", "type": "number", "description": "Brightness multiplier (e.g. 1.2 for +20%).", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        factor = max(0.3, float(p["factor"]))  # <0.3 produces nearly-black image
        out = _output_path(p["image_path"], f"_brightness_{factor}")
        ImageEnhance.Brightness(Image.open(p["image_path"])).enhance(factor).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_adjust_contrast")
class ImageAdjustContrast(BaseTool):
    description = "Adjust contrast. factor > 1 = more contrast."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "factor", "type": "number", "description": "Contrast multiplier.", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        factor = max(0.3, float(p["factor"]))  # <0.3 produces nearly-grey image
        out = _output_path(p["image_path"], f"_contrast_{factor}")
        ImageEnhance.Contrast(Image.open(p["image_path"])).enhance(factor).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_adjust_saturation")
class ImageAdjustSaturation(BaseTool):
    description = "Adjust colour saturation. factor=1 → no change, factor>1 → vivid, factor<1 → desaturated."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "factor", "type": "number", "description": "Saturation multiplier.", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        factor = max(0.3, float(p["factor"]))  # <0.3 produces nearly-greyscale image
        out = _output_path(p["image_path"], f"_saturation_{factor}")
        ImageEnhance.Color(Image.open(p["image_path"])).enhance(factor).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_adjust_sharpness")
class ImageAdjustSharpness(BaseTool):
    description = "Adjust sharpness. factor=1 → no change, factor>1 → sharper, factor<1 → blurrier."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "factor", "type": "number", "description": "Sharpness multiplier.", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        factor = max(0.3, float(p["factor"]))  # <0.3 produces heavily blurred image
        out = _output_path(p["image_path"], f"_sharpness_{factor}")
        ImageEnhance.Sharpness(Image.open(p["image_path"])).enhance(factor).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_apply_blur")
class ImageApplyBlur(BaseTool):
    description = "Apply a Gaussian blur."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "radius", "type": "number", "description": "Blur radius (default 2).", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        radius = p.get("radius", 2)
        out = _output_path(p["image_path"], f"_blur_{radius}")
        Image.open(p["image_path"]).filter(ImageFilter.GaussianBlur(radius=radius)).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_convert_grayscale")
class ImageConvertGrayscale(BaseTool):
    description = "Convert an image to grayscale."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        out = _output_path(p["image_path"], "_grayscale")
        Image.open(p["image_path"]).convert("L").convert("RGB").save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_apply_sepia")
class ImageApplySepia(BaseTool):
    description = "Apply a sepia (warm vintage) tone."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        arr = np.array(Image.open(p["image_path"]).convert("RGB"), dtype=np.float32)
        r = np.clip(arr[:,:,0]*0.393 + arr[:,:,1]*0.769 + arr[:,:,2]*0.189, 0, 255)
        g = np.clip(arr[:,:,0]*0.349 + arr[:,:,1]*0.686 + arr[:,:,2]*0.168, 0, 255)
        b = np.clip(arr[:,:,0]*0.272 + arr[:,:,1]*0.534 + arr[:,:,2]*0.131, 0, 255)
        out = _output_path(p["image_path"], "_sepia")
        Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8)).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_apply_vignette")
class ImageApplyVignette(BaseTool):
    description = "Apply a dark vignette (darkened edges) effect."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "strength", "type": "number", "description": "Vignette strength 0–1 (default 0.5).", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        strength = p.get("strength", 0.5)
        img = Image.open(p["image_path"]).convert("RGB")
        W, H = img.size
        xv, yv = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        mask = 1.0 - strength * np.clip(xv**2 + yv**2, 0, 1)
        result = np.clip(np.array(img, dtype=np.float32) * mask[:, :, np.newaxis], 0, 255).astype(np.uint8)
        out = _output_path(p["image_path"], "_vignette")
        Image.fromarray(result).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_add_watermark")
class ImageAddWatermark(BaseTool):
    description = "Add a semi-transparent text watermark."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "text", "type": "string", "description": "Watermark text.", "required": True},
        {"name": "opacity", "type": "number", "description": "Opacity 0–1 (default 0.3).", "required": False},
        {"name": "position", "type": "string",
         "description": '"center", "bottom-right", or "bottom-left" (default "center").', "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        img = Image.open(p["image_path"]).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        W, H = img.size
        alpha = int(p.get("opacity", 0.3) * 255)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(20, W // 20))
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), p["text"], font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pos = p.get("position", "center")
        # Robustness: LLM may send a list [x, y] instead of enum string
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            xy = (int(pos[0]), int(pos[1]))
        else:
            xy = {"bottom-right": (W - tw - 20, H - th - 20),
                  "bottom-left":  (20, H - th - 20)}.get(pos, ((W - tw) // 2, (H - th) // 2))
        draw.text(xy, p["text"], font=font, fill=(255, 255, 255, alpha))
        out = _output_path(p["image_path"], "_watermark")
        Image.alpha_composite(img, overlay).convert("RGB").save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_add_border")
class ImageAddBorder(BaseTool):
    description = "Add a solid-colour border around an image."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "width", "type": "integer", "description": "Border width in pixels (default 10).", "required": False},
        {"name": "color", "type": "string", "description": 'Border colour hex or name (default "black").', "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        out = _output_path(p["image_path"], f"_border{p.get('width', 10)}")
        ImageOps.expand(Image.open(p["image_path"]).convert("RGB"),
                        border=p.get("width", 10), fill=p.get("color", "black")).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_compress")
class ImageCompress(BaseTool):
    description = "Re-save an image at reduced JPEG quality to shrink file size."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "quality", "type": "integer", "description": "JPEG quality 1–95 (default 75).", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        quality = p.get("quality", 75)
        base = os.path.splitext(p["image_path"])[0]
        out = f"{base}_compressed_q{quality}.jpg"
        Image.open(p["image_path"]).convert("RGB").save(out, "JPEG", quality=quality)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_convert_format")
class ImageConvertFormat(BaseTool):
    description = 'Convert an image to a different format: "PNG", "JPEG", "WEBP", "BMP", "TIFF".'
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "format", "type": "string", "description": "Target format.", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        fmt = p["format"].upper()
        ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp",
               "BMP": ".bmp", "TIFF": ".tif"}.get(fmt, f".{fmt.lower()}")
        out = os.path.splitext(p["image_path"])[0] + "_converted" + ext
        Image.open(p["image_path"]).convert("RGB").save(out, fmt)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_thumbnail")
class ImageThumbnail(BaseTool):
    description = "Create a thumbnail (max dimension in pixels, aspect ratio preserved)."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "max_size", "type": "integer", "description": "Maximum dimension (default 128).", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        size = p.get("max_size", 128)
        img = Image.open(p["image_path"])
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        out = _output_path(p["image_path"], f"_thumbnail{size}")
        img.save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_blend")
class ImageBlend(BaseTool):
    description = "Blend two images. alpha=0 → only base, alpha=1 → only overlay."
    parameters = [
        {"name": "image_path",   "type": "string", "description": "Path to the base image.",    "required": True},
        {"name": "overlay_path", "type": "string", "description": "Path to the overlay image.", "required": True},
        {"name": "alpha", "type": "number", "description": "Overlay weight 0–1 (default 0.5).", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        alpha = p.get("alpha", 0.5)
        base = Image.open(p["image_path"]).convert("RGB")
        overlay = Image.open(p["overlay_path"]).convert("RGB").resize(base.size, Image.Resampling.LANCZOS)
        out = _output_path(p["image_path"], f"_blend_{alpha}")
        Image.blend(base, overlay, alpha).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_extract_edges")
class ImageExtractEdges(BaseTool):
    description = "Extract edges using the FIND_EDGES filter."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        out = _output_path(p["image_path"], "_edges")
        Image.open(p["image_path"]).convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB").save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


@register_tool("image_adjust_hue")
class ImageAdjustHue(BaseTool):
    description = "Shift the hue by a given number of degrees (0–360)."
    parameters = [
        {"name": "image_path", "type": "string", "description": "Input image path.", "required": True},
        {"name": "shift", "type": "number", "description": "Hue shift in degrees.", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json5.loads(params)
        shift = p["shift"] / 360.0
        img = Image.open(p["image_path"]).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        h, w, _ = arr.shape
        flat = arr.reshape(-1, 3)
        shifted = [colorsys.hsv_to_rgb(*((colorsys.rgb_to_hsv(*rgb)[0] + shift) % 1.0,
                                         *colorsys.rgb_to_hsv(*rgb)[1:]))
                   for rgb in flat]
        result = (np.array(shifted, dtype=np.float32).reshape(h, w, 3) * 255).astype(np.uint8)
        out = _output_path(p["image_path"], f"_hue_{int(p['shift'])}")
        Image.fromarray(result).save(out)
        return json5.dumps({"status": "success", "output_path": out}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# ⑤ Tool registry passed to Assistant
# ---------------------------------------------------------------------------

TOOL_LIST = [
    # Main red-team tool set: 16 non-destructive, non-generative edits.
    # Content-removing/high-degradation tools remain registered above for legacy
    # use, but are intentionally excluded from this main list.
    "image_rotate", "image_flip", "image_resize", "image_thumbnail",
    "image_adjust_brightness", "image_adjust_contrast", "image_adjust_saturation",
    "image_adjust_hue", "image_convert_grayscale", "image_apply_sepia",
    "image_adjust_sharpness", "image_apply_vignette", "image_compress",
    "image_convert_format", "image_add_watermark", "image_add_border",
]

# ---------------------------------------------------------------------------
# ⑥ Agent class
# ---------------------------------------------------------------------------

class ImageEditAgent:
    """Stateful, multi-turn image editing agent.

    State transitions:
        __init__
          └─ start_new_conversation()   creates workspace dir, resets image list
               └─ process_request()     generator: yields status dicts while running
                    └─ ...              can be called repeatedly in the same session
    """

    def __init__(self, workspace_root: str = "./workspace"):
        self.workspace_root = Path(workspace_root)
        self.cache_dir = self.workspace_root / "conversation_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._bot: Optional[Assistant] = None  # lazy-initialised

        # Session state
        self.conversation_id: Optional[str] = None
        self.conversation_date: Optional[str] = None
        self.conversation_dir: Optional[Path] = None
        self.history: List[Dict] = []    # full log entries
        self.current_images: List[str] = []  # paths visible in this session

        self.start_new_conversation()

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_bot(self) -> None:
        """Lazily create the qwen-agent Assistant on first use."""
        if self._bot is not None:
            return
        cfg = VLM_CONFIG if USE_VLM else LLM_CONFIG
        self._bot = Assistant(
            llm=cfg,
            system_message=SYSTEM_PROMPT,
            function_list=TOOL_LIST,
        )
        print("[agent] Initialised.")

    def start_new_conversation(self) -> str:
        """Reset session state and create a fresh workspace directory.

        Layout:
            workspace/conversation_cache/
              └─ YYYYMMDD/
                   └─ HHMMSS_<uuid8>/
                        ├─ images/
                        └─ log.json
        """
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        conv_id = f"{now.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}"

        self.conversation_id = conv_id
        self.conversation_date = date_str
        self.current_images = []
        self.history = []

        conv_dir = self.cache_dir / date_str / conv_id
        conv_dir.mkdir(parents=True, exist_ok=True)
        self.conversation_dir = conv_dir
        print(f"[agent] Session: {conv_dir}")
        return conv_id

    # ── Image management ────────────────────────────────────────────────────

    def add_images(self, file_paths: List[str]) -> List[str]:
        """Copy *file_paths* into the session workspace and register them.

        Returns the list of absolute destination paths.
        """
        images_dir = self.conversation_dir / "images"
        images_dir.mkdir(exist_ok=True)
        added: List[str] = []
        for src in file_paths or []:
            try:
                dst = str((images_dir / Path(src).name).resolve())
                shutil.copy2(src, dst)
                if dst not in self.current_images:
                    self.current_images.append(dst)
                added.append(dst)
                print(f"[agent] Registered: {dst}")
            except Exception as exc:
                print(f"[agent] Failed to copy {src}: {exc}")
        return added

    # ── Core processing ─────────────────────────────────────────────────────

    def process_request(self, user_input: str, selected_image: Optional[str] = None):
        """Run the agent for one user turn.

        This is a *generator*.  Each iteration yields a dict:
            status          "processing" | "completed" | "error"
            text            accumulated assistant text so far
            new_images      list of newly produced image paths (this turn)
            tool_calls      number of tool calls executed so far

        The caller (CLI or any other front-end) drives the loop.
        """
        self._init_bot()

        images_dir = self.conversation_dir / "images"
        images_dir.mkdir(exist_ok=True)

        messages = self._build_messages(user_input, selected_image)

        # ── Streaming loop ───────────────────────────────────────────────────
        text = ""
        new_images: List[str] = []
        seen: set = set()
        tool_calls = 0
        t0 = time.time()

        try:
            for responses in self._bot.run(
                messages=messages,
                conversation_images_dir=str(images_dir.resolve()),
            ):
                if not responses:
                    continue

                # Count tool calls
                tc = sum(1 for r in responses if r.get(ROLE) == FUNCTION)
                if tc > tool_calls:
                    tool_calls = tc

                # Extract assistant text (skip raw JSON blobs)
                text = ""
                for rsp in convert_fncall_to_text(responses):
                    if rsp.get(ROLE) == ASSISTANT and rsp.get(CONTENT):
                        c = rsp[CONTENT]
                        if isinstance(c, str) and not (c.startswith("{") and c.endswith("}")):
                            text += c + "\n"
                        elif isinstance(c, list):
                            for item in c:
                                if isinstance(item, dict) and "text" in item:
                                    text += item["text"] + "\n"

                # Extract image paths from tool return values (FUNCTION role)
                for idx, rsp in enumerate(responses):
                    rid = f"{idx}_{rsp.get(ROLE)}_{rsp.get(NAME, '')}"
                    if rid in seen:
                        continue
                    if rsp.get(ROLE) == FUNCTION and rsp.get(CONTENT):
                        for path in self._extract_paths(rsp[CONTENT]):
                            if path not in new_images:
                                new_images.append(path)
                    seen.add(rid)

                yield {
                    "status": "processing",
                    "text": text.strip(),
                    "new_images": list(new_images),
                    "tool_calls": tool_calls,
                    "elapsed": time.time() - t0,
                }

            # ── Finalise ─────────────────────────────────────────────────────
            for img in new_images:
                if img not in self.current_images:
                    self.current_images.append(img)

            self._append_log(user_input, text, new_images)

            yield {
                "status": "completed",
                "text": text.strip() or f"Done. {len(new_images)} image(s) produced.",
                "new_images": new_images,
                "tool_calls": tool_calls,
                "elapsed": time.time() - t0,
            }

        except Exception as exc:
            import traceback
            traceback.print_exc()
            yield {
                "status": "error",
                "text": f"Error: {exc}",
                "new_images": [],
                "tool_calls": tool_calls,
                "elapsed": time.time() - t0,
            }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_messages(self, user_input: str, selected_image: Optional[str]) -> List[Dict]:
        """Build the message list for this turn.

        VLM mode: attach image data alongside text (model can see pixels).
        LLM mode: include file paths as text (model uses paths for tool calls).
        """
        images = (
            [selected_image]
            if selected_image and selected_image in self.current_images
            else list(self.current_images)
        )

        if USE_VLM and images:
            content: List[Dict] = [{"text": user_input}]
            for idx, path in enumerate(images):
                if not os.path.exists(path):
                    continue
                try:
                    w, h = Image.open(path).size
                except Exception:
                    w, h = 0, 0
                content.append({"text": (
                    f"\n[Image {idx + 1}]\n"
                    f"  absolute_path: {path}\n"
                    f"  size: {w}x{h}px\n"
                    "  Use this path as image_path in tool calls.\n"
                )})
                content.append({"image": path})
            return [{"role": "user", "content": content}]

        # LLM mode — pure text
        if images:
            paths_str = "\n".join(f"  - {p}" for p in images)
            body = f"{user_input}\n\nAvailable images:\n{paths_str}"
        else:
            body = (
                f"{user_input}\n\n"
                "No images loaded. Use t2i_gen to generate one."
            )
        return [{"role": "user", "content": body}]

    def _extract_paths(self, result) -> List[str]:
        """Recursively extract valid image file paths from a tool return value.

        Tool return values are JSON strings.  We parse them, then look for
        keys like "output_path".  Only paths that exist on disk are returned.
        """
        paths: List[str] = []

        if isinstance(result, str):
            for parse in (json5.loads, json.loads):
                try:
                    return self._extract_paths(parse(result))
                except Exception:
                    pass
            if result.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                abs_p = os.path.abspath(result)
                if os.path.exists(abs_p):
                    paths.append(abs_p)
            return paths

        if isinstance(result, dict):
            for key in ("output_path", "image_path", "path", "file_path"):
                v = result.get(key, "")
                if isinstance(v, str) and v.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    abs_p = v if os.path.isabs(v) else os.path.abspath(v)
                    if os.path.exists(abs_p) and abs_p not in paths:
                        paths.append(abs_p)
            return paths

        if isinstance(result, list):
            for item in result:
                paths.extend(self._extract_paths(item))

        return paths

    def _append_log(self, user_input: str, response: str, images: List[str]) -> None:
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "response": response,
            "images": images,
        })
        log_file = self.conversation_dir / "log.json"
        with open(log_file, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# ⑦ Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image Editing Agent — Qwen-Agent educational demo",
    )
    parser.add_argument(
        "--image", "-i",
        metavar="PATH",
        action="append",
        dest="images",
        help="Image file to use (repeatable).",
    )
    parser.add_argument(
        "--prompt", "-p",
        metavar="TEXT",
        required=True,
        help="Instruction to send to the agent.",
    )
    args = parser.parse_args()

    agent = ImageEditAgent()

    if args.images:
        agent.add_images(args.images)

    for update in agent.process_request(args.prompt):
        if update["status"] == "processing":
            print(f"  [tool call #{update['tool_calls']}]  elapsed: {update['elapsed']:.1f}s")
        else:
            print(f"\nagent> {update['text']}")
            for p in update.get("new_images", []):
                print(f"  output: {p}")


if __name__ == "__main__":
    main()


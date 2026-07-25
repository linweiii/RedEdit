"""
falconsai_classifier.py - Falconsai NSFW Image Detection Classifier

Lightweight ViT-based binary classifier (normal / nsfw) fine-tuned from
google/vit-base-patch16-224-in21k.

Model: Falconsai/nsfw_image_detection
Reference: https://huggingface.co/Falconsai/nsfw_image_detection
VRAM: <1 GB (ViT-base, 86M params)
"""

import os
import torch
from PIL import Image
from typing import List, Optional, Union

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)  # imageEdit_Agent/
_DEFAULT_CKPT = os.path.join(_PROJECT_ROOT, "checkpoints", "FalconsaiNSFW")
_HF_MODEL_ID = "Falconsai/nsfw_image_detection"


class FalconsaiNSFWClassifier:
    """
    ViT-based binary NSFW image classifier.

    Output contract (same as all UnsafeBench classifiers):
        classify(image_paths) → torch.Tensor [batch, 2]
            column 0 = safe_prob   (label "normal")
            column 1 = unsafe_prob (label "nsfw")
    """

    def __init__(
        self,
        model_name: str = "falconsai-nsfw",
        device: str = "cuda",
        model_path: Optional[str] = None,
    ):
        self.device = device

        # Resolve model path: explicit > local cache > HuggingFace
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists(_DEFAULT_CKPT):
            self.model_path = _DEFAULT_CKPT
        else:
            self.model_path = _HF_MODEL_ID

        self._load_model()

    # ------------------------------------------------------------------
    #  Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        from transformers import AutoModelForImageClassification, ViTImageProcessor

        print(f"Loading Falconsai NSFW model from {self.model_path}...")

        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
        )

        _is_cuda = self.device.startswith("cuda")
        if _is_cuda:
            self.model = self.model.to(self.device)

        self.model.eval()
        self.processor = ViTImageProcessor.from_pretrained(self.model_path)

        # Discover label mapping dynamically
        id2label = self.model.config.id2label  # e.g. {0: "normal", 1: "nsfw"}
        self._safe_idx = None
        self._unsafe_idx = None
        for idx, label in id2label.items():
            idx = int(idx)
            if label.lower() in ("normal", "safe", "sfw"):
                self._safe_idx = idx
            elif label.lower() in ("nsfw", "unsafe"):
                self._unsafe_idx = idx

        if self._safe_idx is None or self._unsafe_idx is None:
            # Fallback: assume 0=normal, 1=nsfw
            print(f"  [Falconsai] WARNING: Unexpected id2label {id2label}, "
                  f"falling back to 0=safe, 1=unsafe")
            self._safe_idx = 0
            self._unsafe_idx = 1

        print(f"Falconsai NSFW model loaded on {self.device} "
              f"(safe_idx={self._safe_idx}, unsafe_idx={self._unsafe_idx})")

    # ------------------------------------------------------------------
    #  Interface methods (UnsafeBench contract)
    # ------------------------------------------------------------------
    def preprocess_images(self, image_paths: List[str]) -> List[str]:
        """Return paths unchanged; actual preprocessing in classify()."""
        return image_paths

    @torch.no_grad()
    def classify(self, images: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """
        Classify images → [batch, 2] tensor of [safe_prob, unsafe_prob].
        """
        if isinstance(images, torch.Tensor):
            raise ValueError("Falconsai classifier requires image paths, not tensors.")

        results = []
        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

                safe_prob = probs[self._safe_idx].item()
                unsafe_prob = probs[self._unsafe_idx].item()
                results.append([safe_prob, unsafe_prob])
            except Exception as e:
                print(f"Warning: Falconsai failed on {img_path}: {e}. Defaulting to Safe.")
                results.append([1.0, 0.0])

        return torch.tensor(results, dtype=torch.float32)

    # ------------------------------------------------------------------
    #  Standard helpers
    # ------------------------------------------------------------------
    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self


def load_falconsai_classifier(
    model_name: str = "falconsai-nsfw",
    device: str = "cuda",
    **kwargs,
) -> FalconsaiNSFWClassifier:
    """Factory function matching UnsafeBench loader convention."""
    return FalconsaiNSFWClassifier(model_name=model_name, device=device, **kwargs)


if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = FalconsaiNSFWClassifier(device=device)

    if len(sys.argv) > 1:
        logits = clf.classify([sys.argv[1]])
        print(f"safe_prob={logits[0, 0]:.4f}  unsafe_prob={logits[0, 1]:.4f}")
    else:
        print("Usage: python falconsai_classifier.py <image_path>")

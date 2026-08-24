import os
import sys
import json
import torch
import torch.utils
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import model_selection
from pathlib import Path
from datasets import load_dataset
import base64
import io

def list_unsafe_instruction():
    """
    Load and parse unsafe instruction categories from the combined instruction file.
    
    Returns:
        tuple: A tuple containing:
            - categories (list): List of unsafe content categories
            - instructions (list): List of full instruction lines
    """
    script_dir = Path(__file__).parent
    
    # 文件实际位置在 imageEdit_data 目录下
    abs_path = script_dir / "imageEdit_data" / "combined_instruction.txt"
    
    if not abs_path.exists():
        # 如果在预期位置找不到，尝试其他可能的位置
        alt_path = script_dir / "data" / "combined_instruction.txt"
        if alt_path.exists():
            abs_path = alt_path
        else:
            # 作为最后手段，尝试在脚本同级目录查找
            alt_path2 = script_dir / "combined_instruction.txt"
            if alt_path2.exists():
                abs_path = alt_path2
            else:
                raise FileNotFoundError(f"Could not find combined_instruction.txt. Checked: {abs_path}, {alt_path}, {alt_path2}")
    
    instructions = open(abs_path, "r").read().splitlines()
    categories = []
    for line in instructions:
        categories.append(line.split(":")[0])
    return categories, instructions 

CATEGORIES = list_unsafe_instruction()[0]
SOURCES = ["Laion5B", "Lexica"]
SEEDGEN_SOURCE = "SeedGen"


# ---------------------------------------------------------------------------
# Generic helpers: discover sources & categories from any metadata JSONL
# ---------------------------------------------------------------------------

# Mapping: dataset_tag → (root_subdir, list_of_metadata_jsonl_relative_paths)
# Each metadata JSONL must have at least "source" and "category" fields.
_DATASET_META_MAP = {
    "UnsafeBench":      ("UnsafeBench", ["train/images/metadata.jsonl",
                                          "test/images/metadata.jsonl"]),
}


def discover_sources_and_categories(dataset_tag: str):
    """
    Scan the metadata JSONL(s) of a given dataset tag and return the
    (sources, categories) actually present in the data.

    Args:
        dataset_tag: One of the keys in _DATASET_META_MAP, e.g.
                     "UnsafeBench", "SeedGen", "SeedGen_sketch", etc.

    Returns:
        (sources: list[str], categories: list[str])  – both sorted.
    """
    if dataset_tag not in _DATASET_META_MAP:
        raise ValueError(
            f"Unknown dataset_tag '{dataset_tag}'. "
            f"Available: {list(_DATASET_META_MAP.keys())}"
        )
    root_sub, meta_files = _DATASET_META_MAP[dataset_tag]
    base_dir = os.path.join(os.path.dirname(__file__), "data", root_sub)

    sources_set = set()
    categories_set = set()
    for mf in meta_files:
        path = os.path.join(base_dir, mf)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                sources_set.add(item.get("source", dataset_tag))
                categories_set.add(item["category"])

    return sorted(sources_set), sorted(categories_set)


def fetch_dataset_by_source_category(dataset_tag: str, source: str, category: str):
    """
    Unified dataset fetcher. Returns a Dataset with (image_path, label) items.

    For UnsafeBench: merges train+test as before.
    For SeedGen / EnhancedSeedGen (and their sketch/bezier variants):
        uses SeedGenDataset filtered by category.
    """
    if dataset_tag == "UnsafeBench":
        return fetch_merged_UnsafeBench_dataset(source=source, category=category)

    # Parse variant from dataset_tag
    # "SeedGen" / "SeedGen_sketch" / "SeedGen_bezier"
    # "EnhancedSeedGen" / "EnhancedSeedGen_sketch" / "EnhancedSeedGen_bezier"
    if dataset_tag.startswith("EnhancedSeedGen"):
        root_name = "EnhancedSeedGen"
        suffix = dataset_tag[len("EnhancedSeedGen"):].lstrip("_")
    elif dataset_tag.startswith("SeedGen"):
        root_name = "SeedGen"
        suffix = dataset_tag[len("SeedGen"):].lstrip("_")
    else:
        raise ValueError(f"Unsupported dataset_tag: {dataset_tag}")

    variant = suffix if suffix in ("sketch", "bezier") else "images"
    image_root = os.path.join(os.path.dirname(__file__), "data", root_name)
    return SeedGenDataset(image_root=image_root, category=category, variant=variant)

def align_unsafe_categories(classifier_name):
    """
    Align unsafe content categories based on the classifier's capabilities.
    
    Different classifiers support different sets of unsafe content categories.
    This function returns the appropriate category list for each classifier.
    
    Args:
        classifier_name (str): Name of the safety classifier
        
    Returns:
        list: List of unsafe content categories supported by the classifier
    """
    if classifier_name in ["NudeNet", "SD_Filter"]:
        unsafe_categories = ["Sexual"]
    elif classifier_name == "NSFW_Detector":
        unsafe_categories = ["Harassment", "Sexual"]
    elif classifier_name == "MultiHeaded":
        unsafe_categories = ["Sexual", "Violence", "Shocking", "Hate", "Political"]
    elif classifier_name == "Q16":
        unsafe_categories = list_unsafe_instruction()[0]
        unsafe_categories.remove("Spam")
        unsafe_categories.remove("Sexual")
    else:  # vlms
        unsafe_categories, _ = list_unsafe_instruction()
    return unsafe_categories

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    ext = os.path.splitext(image_path)[-1].lower().lstrip(".")  # e.g., "png"
    format = ext.upper() if ext != "jpg" else "JPEG"  # PIL expects "JPEG" not "JPG"
    
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_string

def decode_base64_to_image(base64_string, target_size=-1):
    """
    Decode base64 string to PIL Image.
    
    Args:
        base64_string (str): Base64 encoded image string
        target_size (int): Target size for resizing (default: -1, no resizing)
        
    Returns:
        PIL.Image: Decoded image
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    """
    Decode base64 string and save as image file.
    
    Args:
        base64_string (str): Base64 encoded image string
        image_path (str): Output path for the image file
        target_size (int): Target size for resizing (default: -1, no resizing)
    """
    image = decode_base64_to_image(base64_string, target_size=target_size)
    image.save(image_path)
    
class UnsafeBenchDataset(Dataset):
    """
    Dataset class for loading UnsafeBench data.
    
    This dataset automatically downloads data from HuggingFace if not available locally
    and provides a PyTorch Dataset interface for safety evaluation.
    
    Args:
        image_root (str): Root directory for storing images
        source (str): Data source, either "Lexica" or "Laion5B"
        category (str): Safety category (e.g., "Hate", "Violence", "Sexual")
        partition (str): Data partition, either "train" or "test"
    """
    
    def __init__(self, 
                 image_root="data/UnsafeBench", 
                 source="Lexica", 
                 category="Hate", 
                 partition="train"):
        
        self.label_mapping = {"Safe": 0, "Unsafe": 1}
        self.image_root = image_root
        self.source = source
        self.category = category
        self.partition = partition

        metadata = []

        # Check if images are already downloaded
        images_dir = os.path.join(image_root, partition, "images")
        if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
            pass
        else:
            os.makedirs(images_dir, exist_ok=True)
            print(f"Downloading UnsafeBench {partition} images...")
            self._download_and_save(save_path=images_dir)

        # Load metadata
        metadata_path = os.path.join(images_dir, "metadata.jsonl")
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append(json.loads(line))

        # Filter metadata based on source and category
        self.metadata = [item for item in metadata 
                        if item["source"] == source and item["category"] == category]
        
    def __getitem__(self, idx):
        image_fname = self.metadata[idx]["image_fname"]
        image_fname = os.path.join(self.image_root, self.partition, "images", image_fname)
        label = self.metadata[idx]["label"]
        label = self.label_mapping[label]
        return image_fname, label
    
    def __len__(self):
        return len(self.metadata)
    
    def _download_and_save(self, save_path):
        from datasets import load_dataset, load_from_disk
        import tqdm

        # 优先从本地 arrow 缓存加载（离线模式兼容）
        local_arrow_dir = os.path.join(self.image_root, self.partition)
        if os.path.exists(os.path.join(local_arrow_dir, "state.json")):
            dataset = load_from_disk(local_arrow_dir)
        else:
            dataset = load_dataset("yiting/UnsafeBench", split=self.partition)

        metadata = []
        for idx, item in enumerate(tqdm.tqdm(dataset)):
            image = item["image"]
            # Convert to RGB if not already
            if image.mode == "P":
                image = image.convert("RGBA")
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            image_id = item.get("id", str(idx))
            image_filename = f"{image_id}.png"
            image.save(os.path.join(save_path, image_filename))
            
            metadata.append({
                "image_fname": image_filename,
                "label": item["safety_label"],
                "source": item["source"],
                "category": item["category"]
            })
        with open(os.path.join(save_path, "metadata.jsonl"), "w") as f:
            for item in metadata:
                f.write(json.dumps(item) + "\n")

class SeedGenDataset(Dataset):
    """
    Dataset class for loading SeedGen images (original, sketch, or bezier).
    
    Fully compatible with UnsafeBenchDataset: same __getitem__ return
    signature (image_path, label) and same metadata.jsonl schema.
    
    Supports three variants via the ``variant`` parameter:
        "images"  — original T2I generated images  (data/SeedGen/images/)
        "sketch"  — B/W simplified sketches         (data/SeedGen/sketch/)
        "bezier"  — Bezier hand-drawn renderings     (data/SeedGen/bezier/)
    
    Args:
        image_root (str): SeedGen root directory (default: data/SeedGen)
        category (str): Filter by UnsafeBench category (e.g. "Hate").
                        None means load all categories.
        variant (str): Which image variant to load.
                       "images" (default), "sketch", or "bezier".
    """

    # Map variant name → (subdirectory, metadata file)
    _VARIANT_MAP = {
        "images": ("images", "images/metadata.jsonl"),
        "sketch": ("sketch", "sketch_metadata.jsonl"),
        "bezier": ("bezier", "bezier_metadata.jsonl"),
    }

    def __init__(self, image_root=None, category=None, variant="images"):
        if image_root is None:
            image_root = os.path.join(os.path.dirname(__file__), "data", "SeedGen")

        if variant not in self._VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from: {list(self._VARIANT_MAP.keys())}"
            )

        self.label_mapping = {"Safe": 0, "Unsafe": 1}
        self.image_root = image_root
        self.category = category
        self.variant = variant

        subdir, meta_rel = self._VARIANT_MAP[variant]
        self.images_dir = os.path.join(image_root, subdir)
        metadata_path = os.path.join(image_root, meta_rel)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata not found at {metadata_path}. "
                f"Run the corresponding generation script first."
            )

        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append(json.loads(line))

        if category is not None:
            metadata = [m for m in metadata if m["category"] == category]

        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """Return (image_path, label) — same contract as UnsafeBenchDataset."""
        item = self.metadata[idx]
        image_path = os.path.join(self.images_dir, item["image_fname"])
        label = self.label_mapping.get(item["label"], 1)
        return image_path, label


# Custom dataset classes for SMID, NSFWDataset, MultiHeaded_Dataset, Violence_Dataset, Self-harm_Dataset
class CustomDataset(Dataset):
    """
    Custom dataset class for loading various safety datasets from HuggingFace.
    
    Supports multiple dataset types including SMID, NSFWDataset, MultiHeaded_Dataset,
    Violence_Dataset, and Self-harm_Dataset. Automatically downloads and caches
    datasets locally for faster subsequent access.
    
    Args:
        dataset_name (str): Name of the dataset to load
        save_path (str): Local path for saving/loading dataset files
    """
    
    def __init__(self, dataset_name="SMID", save_path="data"):
        
        image_root = os.path.join(save_path, dataset_name, "images")
        metadata_dir = os.path.join(save_path, dataset_name, f"{dataset_name}.tsv")
            
        # Download metadata if not exists
        if not os.path.exists(metadata_dir):
            from datasets import load_from_disk
            local_arrow_dir = os.path.join(save_path, dataset_name)
            state_file = os.path.join(local_arrow_dir, "state.json")
            if os.path.exists(state_file):
                hf_dataset = load_from_disk(local_arrow_dir)
            else:
                hf_dataset = load_dataset(f"yiting/{dataset_name}", split="train")
            data_df = hf_dataset.to_pandas()
            os.makedirs(os.path.dirname(metadata_dir), exist_ok=True)
            data_df.to_csv(metadata_dir, sep="\t", index=False)
                
        data_df = pd.read_csv(metadata_dir, sep="\t")

        # Download and decode images from base64
        if os.path.exists(image_root) and len(os.listdir(image_root)) > 0:
            pass
        else:
            os.makedirs(image_root, exist_ok=True)
            print(f"Decoding {dataset_name} images from base64...")
            for i, row in data_df.iterrows():
                image_base64 = row["image"]
                # Handle case where image is an index reference
                if image_base64.isdigit():
                    image_base64 = data_df.iloc[int(image_base64)]["image"]
                image_fname = os.path.join(image_root, f"{i}.jpg")
                decode_base64_to_image_file(image_base64, image_fname, target_size=-1)
                
        self.image_root = image_root
        self.data = data_df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row_idx = self.data.iloc[idx]["index"]
        image_fname = os.path.join(self.image_root, f"{row_idx}.jpg")
        label = self.data.iloc[idx]["label"]
        
        return {
            "image_fname": image_fname,
            "label": label
        }
        
def fetch_evaluation_dataset(dataset_name):
    """
    Fetch and load an evaluation dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset to load. Supported datasets:
            - "SMID": Safety in Multimodal Intelligence Dataset
            - "NSFWDataset": Not Safe For Work content dataset
            - "MultiHeaded_Dataset": Multi-head classification dataset
            - "Violence_Dataset": Violence detection dataset
            - "Self-harm_Dataset": Self-harm content detection dataset
            - "UnsafeBench_test" or "UnsafeBench_TEST": UnsafeBench test set
    
    Returns:
        Dataset: PyTorch Dataset object for the specified dataset
    
    Raises:
        ValueError: If the dataset name is not recognized
    """
    base_dir = os.path.dirname(__file__)

    if dataset_name in ["SMID", "NSFWDataset", "MultiHeaded_Dataset", "Violence_Dataset", "Self-harm_Dataset"]:
        dataset = CustomDataset(dataset_name=dataset_name, save_path=os.path.join(base_dir, "data"))
        print(f"Loaded {len(dataset)} items from {dataset_name}")
        return dataset
    
    elif dataset_name == "UnsafeBench_test" or dataset_name == "UnsafeBench_TEST":
            
        image_root = os.path.join(base_dir, "data", "UnsafeBench")
        concat_datasets = []
        for source in SOURCES:
            for category in CATEGORIES:
                dataset = UnsafeBenchDataset(image_root=image_root, source=source, category=category, partition="test")
                concat_datasets.append(dataset)
        dataset = ConcatDataset(concat_datasets)
        print(f"Loaded {len(dataset)} items from UnsafeBench_test")
        return dataset

    elif dataset_name.startswith("SeedGen"):
        # Naming convention:
        #   "SeedGen"               → original, all categories
        #   "SeedGen_Hate"          → original, Hate only
        #   "SeedGen_sketch"        → sketch, all categories
        #   "SeedGen_sketch_Hate"   → sketch, Hate only
        #   "SeedGen_bezier"        → bezier, all categories
        #   "SeedGen_bezier_Hate"   → bezier, Hate only
        image_root = os.path.join(base_dir, "data", "SeedGen")
        parts = dataset_name.split("_", 2)  # ["SeedGen", ...]

        variant = "images"
        category = None
        if len(parts) >= 2:
            if parts[1] in ("sketch", "bezier"):
                variant = parts[1]
                category = parts[2] if len(parts) > 2 else None
            else:
                category = "_".join(parts[1:])  # e.g. "Illegal_activity"

        dataset = SeedGenDataset(image_root=image_root, category=category, variant=variant)
        desc = f"variant={variant}, category={category or 'all'}"
        print(f"Loaded {len(dataset)} items from SeedGen ({desc})")
        return dataset

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
def fetch_merged_UnsafeBench_dataset(source, category):
    """
    Fetch and merge train and test splits of UnsafeBench dataset.
    
    Args:
        source (str): Data source, either "Lexica" or "Laion5B"
        category (str): Safety category (e.g., "Hate", "Violence", "Sexual")
    
    Returns:
        ConcatDataset: Combined train and test dataset
    """
    train_set = UnsafeBenchDataset(source=source, category=category, partition="train")
    test_set = UnsafeBenchDataset(source=source, category=category, partition="test")
    dataset = ConcatDataset([train_set, test_set])
    return dataset

def random_draw_testing_adv_samples(classifier_name, prediction_path, source, K=500, seed=2023):
    """
    Randomly draw K samples from UnsafeBench dataset that were originally 
    successfully predicted by each classifier.
    
    This function is useful for adversarial testing - selecting samples that
    the classifier can correctly classify under normal conditions.
    
    Args:
        classifier_name (str): Name of the classifier
        source (str): Data source ("Lexica", "Laion5B", or "both")
        K (int): Number of samples to draw (default: 500)
        seed (int): Random seed for reproducibility (default: 2023)
    
    Returns:
        Subset: PyTorch Subset containing the randomly selected samples
    
    Raises:
        Exception: If prediction data file is not found
    """
    if source == "both":
        sources = SOURCES
    else:
        sources = [source]
    categories = align_unsafe_categories(classifier_name)
    prediction_data_file = f"{prediction_path}/{classifier_name}.json"
    if os.path.exists(prediction_data_file):
        prediction_data = json.load(open(prediction_data_file, "r"))
    else:
        raise Exception(f"Prediction data file {prediction_data_file} not found. Please run the evaluation script first.")

    concat_dataset = []
    all_images_count = 0
    
    for source in sources:
        for category in categories:
            dataset = fetch_merged_UnsafeBench_dataset(source=source, category=category)
            labels = [int(dataset.__getitem__(i)[1]) for i in range(len(dataset))]
            
            if classifier_name in ["llava-v1.5-7b", "instructblip-7b"]:
                predictions = prediction_data[source][category]["detailed_predictions"][0] # predictions of the first prompt
            else:
                predictions = prediction_data[source][category]["predictions"]
        
            assert len(labels) == len(predictions)
            all_images_count += len(labels)
            labels, predictions = np.array(labels), np.array(predictions)
            attack_indices = np.where(labels == predictions)[0]
            dataset = torch.utils.data.Subset(dataset, attack_indices)
            concat_dataset.append(dataset)
            
    concat_dataset = ConcatDataset(concat_dataset)
    
    torch.manual_seed(seed)
    random.seed(seed)
    
    sample_num = min(len(concat_dataset), K)
    random_indices = random.sample(range(len(concat_dataset)), sample_num)
    final_dataset = torch.utils.data.Subset(concat_dataset, random_indices)
    return final_dataset

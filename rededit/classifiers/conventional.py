import os, sys

# Unified checkpoint directory: all model weights are stored under checkpoints/
# classifiers/conventional.py → project root = parent of classifiers/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)  # imageEdit_Agent/
_CKPT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
OPEN_CLIP_CACHE_DIR = os.path.join(_CKPT_DIR, "open_clip")

# If the open_clip cache directory already exists (weights pre-downloaded), enable offline mode automatically
if os.path.isdir(OPEN_CLIP_CACHE_DIR) and os.listdir(OPEN_CLIP_CACHE_DIR):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from PIL import Image
import open_clip
import pickle
from torch.nn import functional as F
import copy
from transformers import AutoFeatureExtractor
from huggingface_hub import snapshot_download
import math

class Q16(torch.nn.Module):
    def __init__(self, checkpoint_dir="./checkpoints/Q16/prompts.p", device="cuda"):
        super(Q16, self).__init__()
        
        self.device = device
        model_name, pretrained = "ViT-L-14", "openai"
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained, quick_gelu=True, cache_dir=OPEN_CLIP_CACHE_DIR)
        self.clip_model.to(torch.float32).to(self.device)
        
        # download checkpoints if not exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
            print(f"Downloading Q16 checkpoints from Hugging Face Hub to {checkpoint_dir}...")
            snapshot_download(repo_id="yiting/Q16",
                    repo_type="model",
                    local_dir=os.path.dirname(checkpoint_dir))
        
        embeddings = torch.FloatTensor(pickle.load(open(checkpoint_dir, 'rb')))
        self.prompts = torch.nn.Parameter(embeddings).to(self.device)
    
    def preprocess_images(self, image_paths):
        images = [self.preprocess(Image.open(img_path)) for img_path in image_paths]
        images = torch.stack(images)
        return images
    
    @ torch.no_grad()
    def classify(self, x):
        x = x.to(self.device)
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        #similarity = (100.0 * image_feat @ text_feat.T)
        #return similarity.squeeze()
        similarity = (100.0 * image_feat @ text_feat.T)
        
        if similarity.size(1) > 1:
        # 计算安全类别和其他类别之间的相对差距
            safe_similarity = similarity[:, 0:1]  # 假设第一个是安全提示
            unsafe_similarities = similarity[:, 1:]
            
            # 计算平均不安全相似度
            avg_unsafe_sim = torch.mean(unsafe_similarities, dim=1, keepdim=True)
            
            # 使用安全和不安全相似度的差值来确定最终分数
            # 这种方法对细微变化更敏感
            score_diff = avg_unsafe_sim - safe_similarity
            # 将差值映射到[0,1]区间
            # 使用sigmoid函数来平滑映射
            raw_score = torch.sigmoid(score_diff)
            final_unsafe_score = torch.clamp(raw_score, 0.0, 1.0)
        else:
            # 如果只有一个类别，直接使用原始相似度
            final_unsafe_score = torch.sigmoid(similarity)
            final_unsafe_score = torch.clamp(final_unsafe_score, 0.0, 1.0)
        
        safe_score = 1.0 - final_unsafe_score
        result = torch.cat([safe_score, final_unsafe_score], dim=1)
        
        return result

    def forward(self, x):
        x = x.to(self.device)
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_feat @ text_feat.T)
        
        logits = F.softmax(similarity, dim=-1) # add this to adapt adversarial attack
        return logits
    
    @ torch.no_grad()
    def forward_head_only(self, e): # input is image features: [bs, embed_len]
        e /= e.norm(dim=-1, keepdim=True)
        e = e.to(self.device)
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        similarity = (100.0 * e @ text_feat.T)
        logits = F.softmax(similarity, dim=-1)
        return logits
        

class MultiHeadedClassifier(torch.nn.Module):
    def __init__(self, checkpoint_dir="./checkpoints/MultiHeaded", device="cuda"):
        super(MultiHeadedClassifier, self).__init__()
        self.device = device
        model_name, pretrained = "ViT-L-14", "openai"
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained, quick_gelu=True, cache_dir=OPEN_CLIP_CACHE_DIR)
        self.clip_model.to(torch.float32).to(self.device)
        
        self.category_heads = torch.nn.ModuleList()
        
        # download checkpoints if not exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Downloading MultiHeaded checkpoints from Hugging Face Hub to {checkpoint_dir}...")
            snapshot_download(repo_id="yiting/MultiHeaded",
                    repo_type="model",
                    local_dir=checkpoint_dir)
            
        unsafe_contents = ["sexual", "violent", "disturbing", "hateful", "political"]
        for head in unsafe_contents:
            checkpoint_path = os.path.join(checkpoint_dir, head+".pt")
            mlp_head = self._init_mlp_head()
            new_mlp_head = self._load_pretrained_weights(mlp_head, checkpoint_path)
            self.category_heads.append(new_mlp_head)
    
        self.category_heads.to(self.device)

    def _init_mlp_head(self):
        return torch.nn.Sequential(
            torch.nn.Linear(768, 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.BatchNorm1d(384),
            torch.nn.Linear(384, 1)
            )
        
    def _load_pretrained_weights(self, mlp_head, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        new_mlp_head = copy.deepcopy(mlp_head)
        new_mlp_head.load_state_dict(checkpoint)
        return new_mlp_head
                
    def preprocess_images(self, image_paths):
        images = [self.preprocess(Image.open(img_path)) for img_path in image_paths]
        images = torch.stack(images)
        return images
    
    @ torch.no_grad()
    def classify(self, x):
        return self.forward(x)
    
    # to adapt the adversarial attack
    def forward(self, x):
        x = x.to(self.device)
        x = self.clip_model.encode_image(x)
        # nsfw score
        out = torch.stack([torch.nn.Sigmoid()(head(x)) for head in self.category_heads], dim=0).permute(1, 0, 2) # [b_s, 5, 1]
        
        # this is for adversarial attack
        out = torch.max(out, dim=1)[0] # [b_s, 1]
        sfw_score = torch.ones_like(out) - out
        logits = torch.cat([sfw_score, out], dim=-1)
        logits = F.softmax(logits, dim=-1) # [b_s, 2]
        
        return logits
    
    @ torch.no_grad()
    def forward_head_only(self, e):
        e = e.to(self.device)
        # nsfw score
        out = torch.stack([torch.nn.Sigmoid()(head(e)) for head in self.category_heads], dim=0).permute(1, 0, 2) # [b_s, 5, 1]
        max_values, max_indices = torch.max(out, dim=1) # [b_s, 1]
        
        embed_list = []
        for idx, max_idx in enumerate(max_indices):
            head = self.category_heads[max_idx]
            embed_block = torch.nn.Sequential(*list(head.children())[:1])
            embed_list.append(embed_block(e[idx].unsqueeze(0)))
            
        embed = torch.cat(embed_list, dim=0)
        sfw_score = torch.ones_like(max_values) - max_values
        logits = torch.cat([sfw_score, max_values], dim=-1)
        logits = F.softmax(logits, dim=-1) # [b_s, 2]
        
        return embed, logits
    
    
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel
from torch.nn import functional as F


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = torch.nn.functional.normalize(image_embeds)
    normalized_text_embeds = torch.nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        chackpoint_dir = os.path.join(_CKPT_DIR, "CompVis", "stable-diffusion-safety-checker")
        if not os.path.exists(chackpoint_dir):
            os.makedirs(chackpoint_dir, exist_ok=True)
            print(f"Downloading StableDiffusionSafetyChecker checkpoints from Hugging Face Hub to {chackpoint_dir}...")
            snapshot_download(repo_id=safety_model_id,
                    repo_type="model",
                    local_dir=chackpoint_dir)
        
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(chackpoint_dir, local_files_only=True)
        # print("Image encoder of SD Filter:", config.vision_config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = torch.nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = torch.nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = torch.nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = torch.nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    def preprocess_images(self, image_paths):
        images = [Image.open(img_path) for img_path in image_paths]
        safety_checker_input = self.safety_feature_extractor(images, return_tensors="pt") # pixel_values: [1, 3, 224, 224]
        images = safety_checker_input.pixel_values
        return images
    
    @torch.no_grad()
    def classify(self, clip_input):
        clip_input = clip_input.to(self.vision_model.device)
        
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    adjustment = 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]
        
        # use logits
        one_hot_list = [[0, 1] if x else [1, 0] for x in has_nsfw_concepts]
        logits = torch.tensor(one_hot_list)
        return logits

        # 优化：使用连续评分机制，对图像编辑操作更敏感
        # continuous_scores = []
        # for res in result:
        #     # 计算所有概念的违规程度，不只是超过阈值的
        #     all_concept_scores = list(res["concept_scores"].values())
        #     all_special_scores = list(res["special_scores"].values())
            
        #     # 计算超过阈值的违规分数总和
        #     positive_concept_scores = [score for score in all_concept_scores if score > 0]
        #     positive_special_scores = [score for score in all_special_scores if score > 0]
        #     total_positive_violations = sum(positive_concept_scores) + sum(positive_special_scores)
            
        #     # 计算接近阈值的分数（即使没超过阈值，也会影响评分）
        #     all_scores = all_concept_scores + all_special_scores
        #     max_score = max(all_scores) if all_scores else 0  # 最接近违规的分数
            
        #     # 综合评分计算
        #     if total_positive_violations > 0:
        #         # 有实际违规：使用对数函数增强对变化的敏感性
        #         # 这样轻微的违规减少也会在评分上有明显体现
        #         unsafe_score = 0.5 + 0.5 * (1.0 - math.exp(-total_positive_violations * 3.0))
        #     elif max_score > 0:
        #         # 没有违规但接近阈值：线性映射
        #         unsafe_score = 0.1 + 0.4 * max_score  # 假设最大可能接近分数为1.0
        #     else:
        #         # 完全安全或负值：使用较小的基础评分
        #         unsafe_score = max(0.01, 0.05 + max_score * 0.3)  # 基础评分加上接近度影响
            
        #     # 确保评分在[0,1]范围内
        #     unsafe_score = min(1.0, max(0.0, unsafe_score))
        #     safe_score = 1.0 - unsafe_score
        #     continuous_scores.append([safe_score, unsafe_score])
        
        # logits = torch.tensor(continuous_scores, dtype=torch.float32)
        # return logits

    # prepare for adversarial attack
    def forward(self, clip_input):
        # Forward pass through the vision model
        clip_input = clip_input.to(self.vision_model.device)
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # Calculate cosine similarities (not distances as the names suggest)
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        # Adjustment parameters
        adjustment = 0.0

        # Special care scores calculation
        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        # Concept scores calculation
        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment

        # Now, derive safe and unsafe logits
        # Unsafe score is the maximum score among all concepts (indicating the strongest match to an "unsafe" concept)
        unsafe_scores = torch.max(concept_scores, dim=1).values
        
        # Safe score can be the complement of the unsafe score
        safe_scores = -unsafe_scores
        
        logits = torch.stack([safe_scores, unsafe_scores], dim=1)
        logits = F.softmax(logits, dim=-1)
        return logits

def load_torch_model(checkpoint_dir):
    
    class Normalization(torch.nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.register_buffer('mean', torch.zeros(shape))
            self.register_buffer('variance', torch.ones(shape))

        def forward(self, x):
            return (x - self.mean) / self.variance.sqrt()

    class NSFWModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = Normalization([768])
            self.linear_1 = torch.nn.Linear(768, 64)
            self.linear_2 = torch.nn.Linear(64, 512)
            self.linear_3 = torch.nn.Linear(512, 256)
            self.linear_4 = torch.nn.Linear(256, 1)
            self.act = torch.nn.ReLU()
            self.act_out = torch.nn.Sigmoid()

        def forward(self, x):
            x = self.norm(x)
            x = self.act(self.linear_1(x))
            x = self.act(self.linear_2(x))
            x = self.act(self.linear_3(x))
            x = self.act_out(self.linear_4(x))
            return x
    
    model = NSFWModel()
    model.load_state_dict(torch.load(checkpoint_dir))
    return model

class NSFWDetector(torch.nn.Module):
    def __init__(self, checkpoint_dir="./checkpoints/NSFWDetector/clip_autokeras_binary_nsfw.pth", device="cuda"):
        super(NSFWDetector, self).__init__()
        
        self.device = device
        model_name, pretrained = "ViT-L-14", "openai"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained, quick_gelu=True, cache_dir=OPEN_CLIP_CACHE_DIR)
        # self.safety_head = NSFW_detector.load_safety_model() # this is tensorflow model
        
        # download checkpoints if not exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
            print(f"Downloading NSFWDetector checkpoints from Hugging Face Hub to {checkpoint_dir}...")
            snapshot_download(repo_id="yiting/NSFWDetector",
                    repo_type="model",
                    local_dir=os.path.dirname(checkpoint_dir))
        
        self.safety_head = load_torch_model(checkpoint_dir) # convert the model to torch
        
        self.model = self.model.to(self.device)
        self.safety_head = self.safety_head.to(self.device)
        
    def preprocess_images(self, image_paths):
        images = [self.preprocess(Image.open(img_path)) for img_path in image_paths]
        images = torch.stack(images)
        return images
    
    @torch.no_grad()
    def classify(self, image_tensor):
        return self.forward(image_tensor)
        
    def forward(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        embeddings = self.model.encode_image(image_tensor)
        embeddings = embeddings / torch.norm(embeddings, keepdim=True, dim=-1)
        nsfw_values = self.safety_head(embeddings)
        sfw_values = torch.ones_like(nsfw_values) - nsfw_values
        logits = torch.cat([sfw_values, nsfw_values], dim=1)
        logits = F.softmax(logits, dim=-1)
        return logits
    
    @torch.no_grad()
    def forward_head_only(self, e):
        e = e.to(self.device)
        e = e/e.norm(keepdim=True, dim=-1)
        x = self.safety_head.norm(e)
        x_embed = self.safety_head.act(self.safety_head.linear_1(x))
        x_embed = self.safety_head.linear_2(x_embed)
        # x_embed = x
        
        nsfw_values = self.safety_head(e)
        sfw_values = torch.ones_like(nsfw_values) - nsfw_values
        logits = torch.cat([sfw_values, nsfw_values], dim=1)
        logits = F.softmax(logits, dim=-1)
        return x_embed, logits

import numpy as np


def _lazy_import_tf():
    """Lazily import tensorflow and keras to avoid protobuf version conflicts."""
    import keras
    import keras.utils as ku
    import tensorflow as tf
    return keras, ku, tf


def load_images(image_paths, image_size):
    '''
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process
    
    '''
    _, ku, _ = _lazy_import_tf()
    loaded_images = []
    loaded_image_paths = []

    for i, img_path in enumerate(image_paths):
        try:
            image = ku.load_img(img_path, target_size = image_size)
            image = ku.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print(i, img_path, ex)
    image_tensor = np.asarray(loaded_images)
    return image_tensor

class NudeNet():
    '''
        Class for loading model and running predictions.
        For example on how to use take a look the if __name__ == '__main__' part.
    '''
    nsfw_model = None

    def __init__(self, model_path=os.path.join(_CKPT_DIR, "NudeNet", "classifier_model.h5")):
        '''
            model = Classifier()
        '''
        _, _, tf = _lazy_import_tf()
        
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Downloading NudeNet checkpoints from Hugging Face Hub to {model_path}...")
            snapshot_download(repo_id="yiting/NudeNet",
                    repo_type="model",
                    local_dir=os.path.dirname(model_path))

        with tf.device('/CPU:0'):
            NudeNet.nsfw_model = tf.keras.models.load_model(model_path, compile=False)
        self.training = False

    # 用于修改模型配置，用于兼容Keras 3。之前的 lr 在 Keras 3 中被改为了 learning_rate
    # def load_old_model(self, path, tf):
    #     try:
    #         # 1. 直接尝试加载（部分 Keras 3 环境可能已经处理了兼容性）
    #         return tf.keras.models.load_model(path)
    #     except Exception:
    #         # 2. 如果失败，手动修改内存中的配置
    #         import h5py
    #         import json
    #         with h5py.File(path, 'r') as f:
    #             # 读取模型配置字符串
    #             model_config = f.attrs.get('model_config')
    #             if model_config is None:
    #                 raise ValueError("无法在 H5 文件中找到 model_config")
                
    #             # 将字节转为字符串并解析为字典
    #             config_dict = json.loads(model_config.decode('utf-8') if isinstance(model_config, bytes) else model_config)
                
    #             # 递归替换所有的 'lr' 为 'learning_rate'
    #             def rename_lr(obj):
    #                 if isinstance(obj, dict):
    #                     if 'lr' in obj:
    #                         obj['learning_rate'] = obj.pop('lr')
    #                     for k, v in obj.items():
    #                         rename_lr(v)
    #                 elif isinstance(obj, list):
    #                     for item in obj:
    #                         rename_lr(item)
                
    #             rename_lr(config_dict)
                
    #             # 使用修改后的配置重建模型结构
    #             # model = tf.keras.models.model_from_config(config_dict)
    #             model = tf.keras.layers.deserialize(config_dict)
                
    #             # 加载权重
    #             model.load_weights(path)
    #             return model

    def preprocess_images(self, image_paths, image_size = (256, 256)):
        _, ku, tf = _lazy_import_tf()
        
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        loaded_images = []
        for i, img_path in enumerate(image_paths):
            try:
                image = ku.load_img(img_path, target_size = image_size)
                image = ku.img_to_array(image)
                image /= 255
                loaded_images.append(image)
            except Exception as ex:
                print(i, img_path, ex)
        image_tensor = np.asarray(loaded_images)
        image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
        return image_tensor
    
    
    def classify(self, image_tensor, batch_size = 32, image_size = (256, 256), categories = ['unsafe', 'safe']):
        '''
            inputs:
                image_paths: list of image paths or can be a string too (for single image)
                batch_size: batch_size for running predictions
                image_size: size to which the image needs to be resized
                categories: since the model predicts numbers, categories is the list of actual names of categories
        '''
        _, _, tf = _lazy_import_tf()
        batch_size = image_tensor.shape[0]
        with tf.device('/CPU:0'):
            image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
            model_preds = NudeNet.nsfw_model.predict(image_tensor, batch_size = batch_size) # output of foolbox attack
            model_preds = torch.tensor(model_preds)
            
            model_preds_swapped = model_preds.clone()
            model_preds_swapped[:, [0, 1]] = model_preds[:, [1, 0]]
        return model_preds_swapped
    
    
    def train(self):
        pass
    
    def eval(self):
        pass
    
    def __call__(self, *args):
        return self.classify(*args)

def load_conventional_classifier(classifier_name, device):
    def _freeze_torch_module(m):
        for p in m.parameters():
            p.requires_grad = False

    # Normalize "cuda" → "cuda:0" to avoid multi-GPU device mismatch
    if device == "cuda":
        device = "cuda:0"

    if classifier_name == "Q16":
        ckpt = os.path.join(_CKPT_DIR, "Q16", "prompts.p")
        classifier = Q16(checkpoint_dir=ckpt, device=device)
    elif classifier_name == "MultiHeaded":
        ckpt = os.path.join(_CKPT_DIR, "MultiHeaded")
        classifier = MultiHeadedClassifier(checkpoint_dir=ckpt, device=device)
    elif classifier_name == "SD_Filter":
        chackpoint_dir = os.path.join(_CKPT_DIR, "CompVis", "stable-diffusion-safety-checker")
        classifier = StableDiffusionSafetyChecker.from_pretrained(chackpoint_dir, local_files_only=True)
    elif classifier_name == "NSFW_Detector":
        ckpt = os.path.join(_CKPT_DIR, "NSFWDetector", "clip_autokeras_binary_nsfw.pth")
        classifier = NSFWDetector(checkpoint_dir=ckpt, device=device)
    elif classifier_name == "NudeNet":
        classifier = NudeNet()
        # Keras model: mark not trainable
        try:
            NudeNet.nsfw_model.trainable = False
        except Exception:
            pass
        return classifier

    # Move to device, set eval mode and freeze grads for torch modules
    if isinstance(classifier, torch.nn.Module):
        try:
            classifier = classifier.to(device)
        except Exception:
            pass
        classifier.eval()
        _freeze_torch_module(classifier)

    return classifier
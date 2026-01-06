# 文件名: sem_seg_dataset.py (专职版本)

import os
import cv2
import torch
import random
import glob
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

# 假设这两个列表在 utils.utils 文件中
from .utils import ANSWER_LIST, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, SHORT_QUESTION_LIST

def init_museg(base_image_dir):
    # (使用我们之前验证过的、最健壮的 init_museg 函数)
    museg_root = os.path.join(base_image_dir, "museg")
    images_root = os.path.join(museg_root, "images") # 你的目录是 images (小写)
    masks_root = os.path.join(museg_root, "masks")   # 你的目录是 masks (小写)

    if not os.path.isdir(images_root) or not os.path.isdir(masks_root):
        print(f"[ERROR] in init_museg: Cannot find 'images' or 'masks' in '{museg_root}'")
        return np.array([]), [], []

    classes_file = os.path.join(museg_root, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    else:
        classes = ["background", "person", "cable", "tube", "indicator", "electrical equipment", "electronic equipment", "mining equipment", "rail area", "support equipment", "door", "tools and materials", "rescue equipment", "container", "metal fixture", "anchoring equipment"]
    classes = np.array(classes)

    all_image_paths = []
    for dirpath, _, filenames in os.walk(images_root):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(dirpath, filename))

    images = []
    labels = []
    for img_path in sorted(all_image_paths):
        relative_path = os.path.relpath(img_path, images_root)
        stem = os.path.splitext(relative_path)[0]
        mask_path = os.path.join(masks_root, stem + ".png")
        if os.path.exists(mask_path):
            images.append(img_path)
            labels.append(mask_path)
            
    print(f"Initialized 'museg' dataset: Found {len(images)} matching image-label pairs.")
    return classes, images, labels


class SemSegDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500, # 可以设一个默认值
        precision: str = "fp16",
        image_size: int = 1024,
        num_classes_per_sample: int = 3,
        use_mm_start_end=True,
        exclude_val: bool = False,
        # --- 关键修改: 移除了 sem_seg_data，因为我们只处理 museg ---
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.use_mm_start_end = use_mm_start_end # 保存为实例属性
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # --- 关键修改: 直接调用 init_museg ---
        self.classes, self.images, self.labels = init_museg(base_image_dir)
        if exclude_val:
            keep_images, keep_labels = [], []
            for img_p, lab_p in zip(self.images, self.labels):
                if ("/val/" in img_p or "\\val\\" in img_p) or ("/val/" in lab_p or "\\val\\" in lab_p):
                    continue
                keep_images.append(img_p)
                keep_labels.append(lab_p)
            self.images, self.labels = keep_images, keep_labels
        if len(self.images) == 0:
            raise ValueError("No images found for MUSEG dataset. Check paths.")

    def __len__(self):
        # 返回真实的数据集长度，而不是一个固定的 epoch 长度
        return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # --- 关键修改: 整个 __getitem__ 只保留 museg 的逻辑 ---
        
        # 使用传入的 idx，而不是随机生成
        image_path = self.images[idx]
        label_path = self.labels[idx]

        label = Image.open(label_path)
        label = np.array(label)
        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        
        unique_label = np.unique(label).tolist()
        if self.ignore_label in unique_label:
            unique_label.remove(self.ignore_label)
        if 0 in unique_label:
            try: unique_label.remove(0)
            except ValueError: pass

        if len(unique_label) == 0:
            # 如果没有前景，可以随机取另一个样本
            return self.__getitem__(random.randint(0, len(self) - 1))

        classes = [self.classes[class_id] for class_id in unique_label]
        if len(classes) >= self.num_classes_per_sample:
            sampled_classes = np.random.choice(classes, size=self.num_classes_per_sample, replace=False).tolist()
        else:
            sampled_classes = classes

        questions, answers, class_ids = [], [], []
        conversations = []
        conv = conversation_lib.default_conversation.copy()

        # 决定使用哪种图像占位符
        image_token_str = DEFAULT_IMAGE_TOKEN

        for sampled_cls in sampled_classes:
            # 1. 生成纯文本的问题
            question_template = random.choice(self.short_question_list)
            question_text = question_template.format(class_name=sampled_cls.lower())
            
            # 2. 构造包含图像占位符的完整 prompt
            #    LLaVA 的标准格式是 <IMAGE_TOKEN>\n<QUESTION>
            full_question_prompt = image_token_str + '\n' + question_text
            
            # 3. 随机选择一个标准回答
            answer = random.choice(self.answer_list)

            # 4. 使用 conversation 模板生成最终的对话字符串
            conv.messages = []
            conv.append_message(conv.roles[0], full_question_prompt)
            conv.append_message(conv.roles[1], answer)
            conversations.append(conv.get_prompt())
            
            # 5. 保存元数据 (如果需要的话)
            questions.append(question_text) # 保存不带占位符的纯问题文本
            answers.append(answer)
            class_ids.append(self.classes.tolist().index(sampled_cls))
            
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        
        label = torch.from_numpy(label).long()
        masks = []
        for class_id in class_ids:
            masks.append(label == class_id)
        masks = torch.stack(masks, dim=0)
            
        # 返回固定的9元素元组
        return (
            image_path, image, image_clip, conversations,
            masks, label, resize, questions, sampled_classes
        )

class MuSegValDataset(torch.utils.data.Dataset):
    """MuSeg 的验证数据集：与原始 collate_fn 对齐（10 元组，inference=True）。"""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std  = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size   = 1024
    ignore_label = 255

    def __init__(self, base_image_dir, tokenizer, vision_tower, image_size=1024, use_mm_start_end=True):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.use_mm_start_end = use_mm_start_end

        # 直接复用初始化函数，拿到 classes / 全部路径
        self.classes, all_images, all_labels = init_museg(base_image_dir)

        # 只保留 /val/ 子目录（Windows 也兼容）
        self.images, self.labels = [], []
        for img_p, lab_p in zip(all_images, all_labels):
            if ("/val/" in img_p or "\\val\\" in img_p) and ("/val/" in lab_p or "\\val\\" in lab_p):
                self.images.append(img_p)
                self.labels.append(lab_p)

        if len(self.images) == 0:
            raise ValueError("MuSeg 验证集为空：请检查 dataset/museg/images/val 与 masks/val 是否存在并匹配。")

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]

        # 读图 / 读整幅标签
        label_np = np.array(Image.open(label_path))
        img_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 构造 conversations：每个前景类出一条问句
        unique_label = np.unique(label_np).tolist()
        for remove_v in [self.ignore_label, 0]:
            if remove_v in unique_label:
                try: unique_label.remove(remove_v)
                except ValueError: pass

        # 如果该图没有任何前景，随机换一张（可选）
        if len(unique_label) == 0:
            return self.__getitem__(np.random.randint(0, len(self)))

        class_ids = sorted(unique_label)
        class_names = [self.classes[cid] for cid in class_ids]

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        # 一律只放“裸 <image>”，是否加 <im_start>/<im_end> 交给 collate_fn 统一处理
        image_token_str = DEFAULT_IMAGE_TOKEN

        for cname in class_names:
            question = f"What is {cname.lower()} in this image? Please output segmentation mask."
            conv.messages = []
            conv.append_message(conv.roles[0], image_token_str + "\n" + question)
            conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())

        # 预处理 clip & sam 输入
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_sam  = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam  = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

        # 按 class_ids 生成多张二值 GT 掩码 [K,H,W]（uint8/0-1）
        label = torch.from_numpy(label_np).long()
        masks = []
        for cid in class_ids:
            masks.append((label == cid).to(torch.uint8))
        masks = torch.stack(masks, dim=0)  # [K, H, W]

        # 注意：与原始 collate_fn 对齐的 10 元组；questions/sample_classes 用 None 占位
        inference = True
        return (
            image_path,          # str
            image_sam,           # [3, H', W'] float (已标准化+pad)
            image_clip,          # [3, 224, 224] clip tensor
            conversations,       # List[str]，每类一条问句
            masks,               # [K, H, W] uint8
            label,               # [H, W] 整幅标签，用于原图尺寸等
            resize,              # (H', W')
            None,                # questions（验证用不到）
            None,                # sampled_classes（验证用不到）
            inference,           # True
        )

import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from utils.utils import ANSWER_LIST, SHORT_QUESTION_LIST

# id -> name 映射（矿山主题）
ID_TO_NAME = {
    0: "background",
    1: "person",
    2: "cable",
    3: "tube",
    4: "indicator",
    5: "electrical equipment",
    6: "electronic equipment",
    7: "mining equipment",
    8: "rail area",
    9: "support equipment",
    10: "door",
    11: "tools and materials",
    12: "rescue equipment",
    13: "container",
    14: "metal fixture",
    15: "anchoring equipment"
}

class MusegMineDataset(Dataset):
    def __init__(self, data_root, tokenizer, vision_tower,
                 transform=None, image_size=1024, mode="multi-round",
                 mine=None, use_depth=False):
        if mine is not None:
            self.root = os.path.join(data_root, mine)
        else:
            self.root = data_root

        self.image_dir = os.path.join(self.root, "Image")
        self.mask_dir = os.path.join(self.root, "Label")
        self.depth_dir = os.path.join(self.root, "Depth")
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.use_depth = use_depth

        self.tokenizer = tokenizer
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.samples = self._load_data()
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found. Check {self.image_dir} and {self.mask_dir}")

    def _load_data(self):
        samples = []
        if not os.path.isdir(self.image_dir):
            return samples
        for f in sorted(os.listdir(self.image_dir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(self.image_dir, f)
            stem = os.path.splitext(f)[0]
            mask_path = os.path.join(self.mask_dir, stem + "_label.png")
            if not os.path.exists(mask_path):
                continue
            depth_path = None
            if self.use_depth:
                cand = os.path.join(self.depth_dir, stem + ".png")
                if os.path.exists(cand):
                    depth_path = cand
            samples.append({"image": img_path, "mask": mask_path, "depth": depth_path})
        return samples

    def __len__(self):
        return len(self.samples)

    def _resize_and_pad(self, image, mask):
        h0, w0 = image.shape[:2]
        scale = min(self.image_size / w0, self.image_size / h0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        image_resized = cv2.resize(image, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        if pad_h > 0:
            image_resized = np.concatenate([image_resized, np.zeros((pad_h, new_w, 3), dtype=np.uint8)], axis=0)
            mask_resized = np.concatenate([mask_resized, np.zeros((pad_h, new_w), dtype=np.uint8)], axis=0)
        if pad_w > 0:
            image_resized = np.concatenate([image_resized, np.zeros((self.image_size, pad_w, 3), dtype=np.uint8)], axis=1)
            mask_resized = np.concatenate([mask_resized, np.zeros((self.image_size, pad_w), dtype=np.uint8)], axis=1)

        return image_resized, mask_resized

    def __getitem__(self, idx):
        rec = self.samples[idx]
        image = cv2.imread(rec["image"])[..., ::-1]  # BGR->RGB
        mask = cv2.imread(rec["mask"], cv2.IMREAD_GRAYSCALE)

        depth = None
        if self.use_depth and rec.get("depth"):
            depth = cv2.imread(rec["depth"], cv2.IMREAD_UNCHANGED)

        # CLIP image encoder输入
        image_clip = self.clip_image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # resize + pad
        image_resized, mask_resized = self._resize_and_pad(image, mask)

        # mask tensor
        mask_tensor_chw = torch.from_numpy(mask_resized.astype(np.int64))  # HxW
        mask_tensor_bchw = mask_tensor_chw.unsqueeze(0).float()  # 1xHxW

        # image tensor
        if self.transform:
            # 把 numpy(HWC) 直接给 transform（包含 ToTensor + Normalize）
            image_tensor = self.transform(image_resized)
        else:
            # 手动 /255 + Normalize
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

        # 找到 mask 里存在的类别
        present_ids = sorted(np.unique(mask_resized).tolist())
        present_ids = [cid for cid in present_ids if cid != 0]
        class_names = [ID_TO_NAME.get(cid, "unknown") for cid in present_ids]

        # 生成问答
        questions = []
        conversations = []
        if self.mode == "multi-round":
            for cname in class_names:
                q = random.choice(self.short_question_list).format(class_name=cname.lower())
                a = random.choice(self.answer_list)
                conv = conversation_lib.default_conversation.copy()
                conv.messages = []
                conv.append_message(conv.roles[0], q)
                conv.append_message(conv.roles[1], a)
                conversations.append(conv.get_prompt())
                questions.append(q)
        else:
            if len(class_names) == 0:
                q = "Please segment the relevant objects in the image."
            else:
                joined = ", ".join(class_names)
                q = f"In this mining scene, please segment: {joined}."
            a = random.choice(self.answer_list)
            conv = conversation_lib.default_conversation.copy()
            conv.messages = []
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], a)
            conversations.append(conv.get_prompt())
            questions.append(q)

        return (
            rec["image"],
            image_tensor,
            image_clip,
            conversations,
            mask_tensor_bchw,
            mask_tensor_chw,
            image_resized.shape[:2],
            questions,
            class_names,
            False
        )

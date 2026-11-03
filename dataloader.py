import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

class NewsDataset(Dataset):
    def __init__(self, json_path, tokenizer, image_transform, max_seq_len):
        super().__init__()
        print(f"Loading data from: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_path}")
            self.data = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_path}")
            self.data = []
        except Exception as e:
            print(f"An unexpected error occurred while loading {json_path}: {e}")
            self.data = []

        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_seq_len = max_seq_len
        print(f"Loaded {len(self.data)} data points.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        content = str(item.get('content', ''))
        cs_rationale = str(item.get('cs_rationale', ''))
        td_rationale = str(item.get('td_rationale', ''))
        img_rationale = str(item.get('img_rationale', ''))
        img_desc = str(item.get('forgery_description', ''))

        content_encoding = self.tokenizer(
            content,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        cs_rationale_encoding = self.tokenizer(
            cs_rationale, add_special_tokens=True, max_length=self.max_seq_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        td_rationale_encoding = self.tokenizer(
            td_rationale, add_special_tokens=True, max_length=self.max_seq_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        img_rationale_encoding = self.tokenizer(
            img_rationale, add_special_tokens=True, max_length=self.max_seq_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        img_desc_encoding = self.tokenizer(
            img_desc, add_special_tokens=True, max_length=self.max_seq_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )

        image_path = item.get('image_path')
        if image_path and not os.path.isabs(image_path):
            image_path = str(PROJECT_ROOT / image_path)
        image_tensor = torch.zeros(3, 224, 224)
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.image_transform(image)
            except FileNotFoundError:
                print(f"Warning: Image file not found at {image_path} for index {idx}. Using placeholder.")
            except Exception as e:
                print(f"Warning: Error loading image {image_path} for index {idx}: {e}. Using placeholder.")
        elif image_path:
             print(f"Warning: Image path specified but not found: {image_path} for index {idx}. Using placeholder.")
        else:
             print(f"Warning: Image path not specified for index {idx}. Using placeholder.")


        label = torch.tensor(item.get('label', -1), dtype=torch.long)
        cs_pred = torch.tensor(item.get('cs_pred', -1), dtype=torch.long)
        td_pred = torch.tensor(item.get('td_pred', -1), dtype=torch.long)
        img_pred = torch.tensor(item.get('img_pred', -1), dtype=torch.long)

        if label == -1: print(f"Warning: Ground truth label missing for index {idx}.")
        if cs_pred == -1: print(f"Warning: CS prediction missing for index {idx}.")
        if td_pred == -1: print(f"Warning: TD prediction missing for index {idx}.")
        if img_pred == -1: print(f"Warning: IMG prediction missing for index {idx}.")


        return {
            'content_ids': content_encoding['input_ids'].squeeze(0),
            'content_mask': content_encoding['attention_mask'].squeeze(0),
            'image': image_tensor,
            'label': label,
            'cs_rationale_ids': cs_rationale_encoding['input_ids'].squeeze(0),
            'cs_rationale_mask': cs_rationale_encoding['attention_mask'].squeeze(0),
            'cs_pred': cs_pred,
            'td_rationale_ids': td_rationale_encoding['input_ids'].squeeze(0),
            'td_rationale_mask': td_rationale_encoding['attention_mask'].squeeze(0),
            'td_pred': td_pred,
            'img_rationale_ids': img_rationale_encoding['input_ids'].squeeze(0),
            'img_rationale_mask': img_rationale_encoding['attention_mask'].squeeze(0),
            'img_pred': img_pred,
            'img_desc_ids': img_desc_encoding['input_ids'].squeeze(0),
            'img_desc_mask': img_desc_encoding['attention_mask'].squeeze(0),
            'post_id': item.get('post_id', ''),
            'original_image_path': image_path if image_path else "N/A"
        }

def get_image_transform(image_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

def create_dataloader(json_path, tokenizer, image_transform, max_seq_len, batch_size, shuffle=True, num_workers=4):
    dataset = NewsDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_seq_len=max_seq_len
    )

    if not dataset.data:
         print(f"Dataset from {json_path} is empty. Cannot create DataLoader.")
         return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


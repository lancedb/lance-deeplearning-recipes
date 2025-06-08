
import os
import json
import requests
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import lancedb
import pyarrow as pa
from tqdm import tqdm
import logging
from datetime import datetime
import io
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import cv2
import concurrent.futures
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class COCOConfig:
    """Configuration for COCO dataset processing"""
    COCO_URLS = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'test_images': 'http://images.cocodataset.org/zips/test2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'stuff_annotations': 'http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip'
    }
    DATA_DIR = Path('./coco_data_new')
    LANCEDB_DIR = 'lance_tables'
    BATCH_SIZE = 64
    MAX_TEXT_LENGTH = 77
    CLIP_MODEL = 'openai/clip-vit-base-patch32'
    TEXT_MODEL = 'BAAI/bge-base-en-v1.5'
    STORE_ORIGINAL_IMAGES = True
    STORE_THUMBNAIL_IMAGES = True
    THUMBNAIL_SIZE = (256, 256)
    MAX_IMAGES_PER_SPLIT = None

class COCODatasetComplete(Dataset):
    def __init__(self, data_records: List[Dict], processor, config: COCOConfig):
        self.data_records = data_records
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.data_records)

    def __getitem__(self, idx):
        try:
            record = self.data_records[idx]
            image_path = Path(record['image_path'])
            if not image_path.exists(): return self._get_dummy_item()

            image = Image.open(image_path).convert('RGB')
            
            image_bytes, thumbnail_bytes = None, None
            if self.config.STORE_ORIGINAL_IMAGES:
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=95)
                image_bytes = img_buffer.getvalue()
            if self.config.STORE_THUMBNAIL_IMAGES:
                thumbnail = image.copy()
                thumbnail.thumbnail(self.config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                thumb_buffer = io.BytesIO()
                thumbnail.save(thumb_buffer, format='JPEG', quality=85)
                thumbnail_bytes = thumb_buffer.getvalue()

            caption_for_clip = record.get('captions', [''])[0]
            inputs = self.processor(
                text=caption_for_clip, images=image, return_tensors="pt",
                padding='max_length', truncation=True, max_length=self.config.MAX_TEXT_LENGTH
            )

            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'captions': record.get('captions', []),
                'image_bytes': image_bytes,
                'thumbnail_bytes': thumbnail_bytes,
                'record': record
            }
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            return self._get_dummy_item()

    def _get_dummy_item(self):
        return {
            'pixel_values': torch.zeros((3, 224, 224)),
            'captions': [], 'image_bytes': None, 'thumbnail_bytes': None, 'record': {}
        }

class COCOProcessor:
    def __init__(self):
        self.config = COCOConfig()
        self.setup_directories()
        self.load_models()

    def setup_directories(self):
        Path(self.config.DATA_DIR).mkdir(exist_ok=True)
        Path(self.config.LANCEDB_DIR).mkdir(exist_ok=True)

    def load_models(self):
        logger.info("Loading models...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model = CLIPModel.from_pretrained(self.config.CLIP_MODEL).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.config.CLIP_MODEL)
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.TEXT_MODEL)
        self.text_model = AutoModel.from_pretrained(self.config.TEXT_MODEL).to(self.device)
        logger.info(f"Models loaded on device: {self.device}")

    def _download_and_extract_file(self, url: str, check_path: str):
        """Helper to download a file and extract it if check_path doesn't exist."""
        filename = url.split('/')[-1]
        zip_path = self.config.DATA_DIR / filename
        
        # Download
        if not zip_path.exists():
            logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(desc=filename, total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Extract
        if not (self.config.DATA_DIR / check_path).exists():
            logger.info(f"Extracting {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.config.DATA_DIR)
        else:
            logger.info(f"'{check_path}' found, skipping extraction for {filename}.")
            
    def download_coco_dataset(self):
        logger.info("Starting COCO dataset download and extraction...")
        files_to_process = {
            'annotations': ('annotations', self.config.COCO_URLS['annotations']),
            'stuff_annotations': ('annotations/stuff_train2017.json', self.config.COCO_URLS['stuff_annotations']),
            'train_images': ('train2017', self.config.COCO_URLS['train_images']),
            'val_images': ('val2017', self.config.COCO_URLS['val_images']),
            'test_images': ('test2017', self.config.COCO_URLS['test_images']),
        }
        for name, (check_path, url) in files_to_process.items():
            self._download_and_extract_file(url, check_path)
        logger.info("Dataset download and extraction check complete.")

    def _process_single_image(self, args):
        img_id, img_info, img_dir, coco_instances, coco_stuff, captions_map = args
        image_path = img_dir / img_info['file_name']
        if not image_path.exists(): return None
        image_captions = captions_map.get(img_id, [])
        if not image_captions: return None

        instance_anns = coco_instances.loadAnns(coco_instances.getAnnIds(imgIds=[img_id])) if coco_instances else []
        stuff_anns = coco_stuff.loadAnns(coco_stuff.getAnnIds(imgIds=[img_id])) if coco_stuff else []
        
        bboxes = [ann.get('bbox', []) for ann in instance_anns if ann.get('bbox')]
        
        record = {
            'image_id': img_id, 'image_path': str(image_path), 'file_name': img_info['file_name'],
            'height': img_info['height'], 'width': img_info['width'], 'captions': image_captions,
            'coco_url': img_info.get('coco_url', ''), 'flickr_url': img_info.get('flickr_url', ''),
            'date_captured': img_info.get('date_captured', ''),
            'bounding_boxes': bboxes,

            'object_classes': [coco_instances.cats[ann['category_id']]['name'] for ann in instance_anns if 'category_id' in ann],
            'stuff_classes': [coco_stuff.cats[ann['category_id']]['name'] for ann in stuff_anns if 'category_id' in ann and coco_stuff],
            
            'object_areas': [ann.get('area', 0.0) for ann in instance_anns],
            'instance_masks': [{'segmentation': ann.get('segmentation'), 'category_id': ann.get('category_id')} for ann in instance_anns],
            'stuff_masks': [{'segmentation': ann.get('segmentation'), 'category_id': ann.get('category_id')} for ann in stuff_anns],
            'num_objects': len(bboxes), 'num_instance_masks': len(instance_anns), 'num_stuff_masks': len(stuff_anns),
            'has_objects': len(bboxes) > 0, 'has_segmentation': len(instance_anns) > 0, 'has_stuff': len(stuff_anns) > 0,
        }
        return record

    def load_complete_annotations(self) -> List[Dict[str, Any]]:
        logger.info("Loading COCO annotations (1 row per image)...")
        ann_dir = self.config.DATA_DIR / 'annotations'
        all_records = []
        for split in ['train', 'val']:
            logger.info(f"Processing {split} split...")
            instances_file = ann_dir / f'instances_{split}2017.json'
            captions_file = ann_dir / f'captions_{split}2017.json'
            stuff_file = ann_dir / f'stuff_{split}2017.json'
            img_dir = self.config.DATA_DIR / f'{split}2017'

            if not all([instances_file.exists(), captions_file.exists(), img_dir.exists()]):
                logger.warning(f"Skipping '{split}' due to missing files."); continue
            
            coco_instances = COCO(str(instances_file))
            coco_stuff = COCO(str(stuff_file)) if stuff_file.exists() else None
            with open(captions_file) as f: captions_data = json.load(f)
            
            captions_map = {}
            for ann in captions_data['annotations']:
                captions_map.setdefault(ann['image_id'], []).append(ann['caption'])
            
            images_to_process = list(coco_instances.imgs.items())
            if self.config.MAX_IMAGES_PER_SPLIT:
                images_to_process = images_to_process[:self.config.MAX_IMAGES_PER_SPLIT]
            
            tasks = [(img_id, img_info, img_dir, coco_instances, coco_stuff, captions_map) for img_id, img_info in images_to_process]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(tqdm(executor.map(self._process_single_image, tasks), total=len(tasks), desc=f"Gathering {split} annotations"))
            
            for image_record in results:
                if image_record:
                    image_record['split'] = split
                    all_records.append(image_record)
        logger.info(f"Loaded a total of {len(all_records)} image records.")
        return all_records

    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        texts_with_instruction = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        encoded = self.text_tokenizer(texts_with_instruction, padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.text_model(**encoded)
            token_embeds = output.last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1).expand(token_embeds.size()).float()
            sum_embeds = torch.sum(token_embeds * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_embeds = sum_embeds / sum_mask
        return mean_embeds.cpu().numpy()

    def collate_fn(self, batch):
        valid_batch = [item for item in batch if item.get('record')]
        if not valid_batch: return None
        return {
            'pixel_values': torch.stack([item['pixel_values'] for item in valid_batch]),
            'captions': [item['captions'] for item in valid_batch],
            'image_bytes': [item['image_bytes'] for item in valid_batch],
            'thumbnail_bytes': [item['thumbnail_bytes'] for item in valid_batch],
            'records': [item['record'] for item in valid_batch]
        }

    def process_batch(self, batch) -> List[Dict[str, Any]]:
        if not batch: return []
        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(pixel_values=batch['pixel_values'].to(self.device)).cpu().numpy()
        
        avg_text_embeddings = []
        text_dim = self.text_model.config.hidden_size
        for captions_list in batch['captions']:
            if captions_list:
                caption_embeds = self.get_text_embeddings(captions_list)
                avg_text_embeddings.append(np.mean(caption_embeds, axis=0))
            else:
                avg_text_embeddings.append(np.zeros(text_dim, dtype=np.float32))
        
        batch_data = []
        for i, record in enumerate(batch['records']):
            batch_data.append({
                'image_embedding': image_embeddings[i],
                'text_embedding': avg_text_embeddings[i],
                'image_bytes': batch['image_bytes'][i],
                'thumbnail_bytes': batch['thumbnail_bytes'][i],
                **record
            })
        return batch_data

    def create_lancedb_table(self, db, table_name):
        mask_struct = pa.struct([
            pa.field('segmentation', pa.binary()), pa.field('category_id', pa.int64())
        ])
        
        schema = pa.schema([
            pa.field('image_embedding', pa.list_(pa.float32(), list_size=512)),
            pa.field('text_embedding', pa.list_(pa.float32(), list_size=768)),
            pa.field('image_bytes', pa.binary()),
            pa.field('thumbnail_bytes', pa.binary()),
            pa.field('image_path', pa.string()),
            pa.field('file_name', pa.string()),
            pa.field('height', pa.int32()),
            pa.field('width', pa.int32()),
            pa.field('coco_url', pa.string()),
            pa.field('flickr_url', pa.string()),
            pa.field('date_captured', pa.string()),
            pa.field('captions', pa.list_(pa.string())),
            pa.field('image_id', pa.int64()),
            pa.field('split', pa.string()),
            pa.field('bounding_boxes', pa.list_(pa.list_(pa.float32()))),
            
            pa.field('object_classes', pa.list_(pa.string())),
            pa.field('stuff_classes', pa.list_(pa.string())),

            pa.field('object_areas', pa.list_(pa.float64())),
            pa.field('instance_masks', pa.list_(mask_struct)),
            pa.field('stuff_masks', pa.list_(mask_struct)),
            pa.field('num_objects', pa.int32()),
            pa.field('num_instance_masks', pa.int32()),
            pa.field('num_stuff_masks', pa.int32()),
            pa.field('has_objects', pa.bool_()),
            pa.field('has_segmentation', pa.bool_()),
            pa.field('has_stuff', pa.bool_()),
        ])
        return db.create_table(table_name, schema=schema, mode='overwrite')

    def add_batch_to_table(self, table, processed_data: List[Dict[str, Any]]):
        if not processed_data: return
        
        for item in processed_data:
            # Pickle the raw segmentation objects into bytes just before ingestion
            for mask_obj in item.get('instance_masks', []):
                if mask_obj and 'segmentation' in mask_obj and mask_obj['segmentation'] is not None:
                    mask_obj['segmentation'] = pickle.dumps(mask_obj['segmentation'])
            for mask_obj in item.get('stuff_masks', []):
                if mask_obj and 'segmentation' in mask_obj and mask_obj['segmentation'] is not None:
                    mask_obj['segmentation'] = pickle.dumps(mask_obj['segmentation'])

        table.add(processed_data)

    def run_ingestion(self):
        self.download_coco_dataset()
        all_records = self.load_complete_annotations()
        num_workers=os.cpu_count()
        dataset = COCODatasetComplete(all_records, self.clip_processor, self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            collate_fn=self.collate_fn, num_workers=os.cpu_count(), pin_memory=True,
            prefetch_factor=10 if num_workers > 0 else 2
        )

        logger.info(f"Connecting to LanceDB at: {self.config.LANCEDB_DIR}")
        db = lancedb.connect(self.config.LANCEDB_DIR)
        table_name = 'coco_new' 
        
        if table_name in db.table_names():
            logger.warning(f"Table '{table_name}' already exists. Overwriting.")
        
        table = self.create_lancedb_table(db, table_name)

        for batch in tqdm(dataloader, desc="Processing and Ingesting Batches"):
            if not batch: continue
            processed_batch = self.process_batch(batch)
            self.add_batch_to_table(table, processed_batch)

        logger.info("Completed ingestion.")
        logger.info("Creating vector indices...")
        table.optimize()
        table.create_index(vector_column_name='image_embedding', replace=True)
        table.create_index(vector_column_name='text_embedding', replace=True)
        logger.info("Indexing complete.")

if __name__ == '__main__':
    logger.info("Starting COCO to LanceDB Ingestion Script")
    processor = COCOProcessor()
    processor.run_ingestion()
    logger.info("Script finished successfully.")
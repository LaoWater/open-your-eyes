# Open Images V7 Kitchen Safety Dataset Setup
# Complete pipeline to download, filter, and prepare kitchen safety data

import os
import pandas as pd
import requests
import subprocess
from pathlib import Path
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenImagesKitchenSafetyDataset:
    def __init__(self, base_dir="./openimages_kitchen"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Kitchen and safety related class IDs from Open Images
        self.target_classes = {
            # Kitchen tools and appliances
            '/m/0hkxq': 'Kitchen knife',
            '/m/04ctx': 'Knife',
            '/m/0dt3t': 'Spoon',
            '/m/0cmx8': 'Fork', 
            '/m/04dr76w': 'Mixing bowl',
            '/m/0l515': 'Oven',
            '/m/078n6': 'Stove',
            '/m/0fx9l': 'Refrigerator',
            
            # Hands and protective equipment  
            '/m/0k65p': 'Hand',
            '/m/0174k2': 'Glove',
            '/m/04yx4': 'Clothing',
            '/m/01940j': 'Shirt',
            
            # Containers and potential hazards
            '/m/0440zs': 'Bottle',
            '/m/02p0tk3': 'Human body',
            '/m/01g317': 'Person',
            '/m/083vt': 'Wood',
            '/m/06_fw': 'Metal',
            
            # Kitchen specific items
            '/m/01k6s3': 'Potato',
            '/m/0270h': 'Bread',
            '/m/0cdn1': 'Table',
            '/m/04bcr3': 'Mug',
            '/m/0dt2t': 'Cup'
        }
        
        # Safety scenarios we want to detect
        self.safety_labels = {
            'knife_handling': ['Kitchen knife', 'Knife', 'Hand'],
            'protective_equipment': ['Glove', 'Hand'],
            'hot_surface_contact': ['Hand', 'Stove', 'Oven'],
            'proper_workspace': ['Table', 'Kitchen knife', 'Cutting board']
        }

    def download_metadata(self):
        """Download Open Images V7 metadata files"""
        logger.info("Downloading Open Images V7 metadata...")
        
        metadata_urls = {
            'train_annotations': 'https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-bbox.csv',
            'validation_annotations': 'https://storage.googleapis.com/openimages/v7/oidv7-validation-annotations-bbox.csv',
            'test_annotations': 'https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-bbox.csv',
            'class_descriptions': 'https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv',
            'train_images': 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv',
            'validation_images': 'https://storage.googleapis.com/openimages/v5/validation-images-with-rotation.csv',
            'test_images': 'https://storage.googleapis.com/openimages/v5/test-images-with-rotation.csv'
        }
        
        for name, url in metadata_urls.items():
            filepath = self.base_dir / f"{name}.csv"
            if not filepath.exists():
                logger.info(f"Downloading {name}...")
                response = requests.get(url)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.info(f"✓ Downloaded {name}")
            else:
                logger.info(f"✓ {name} already exists")

    def filter_kitchen_images(self, split='train', max_images=50000):
        """Filter images containing kitchen/safety relevant objects"""
        logger.info(f"Filtering {split} images for kitchen safety content...")
        
        # Load annotations
        annotations_df = pd.read_csv(self.base_dir / f"{split}_annotations.csv")
        
        # Filter for our target classes
        target_class_ids = list(self.target_classes.keys())
        kitchen_annotations = annotations_df[
            annotations_df['LabelName'].isin(target_class_ids)
        ]
        
        # Group by image and count relevant objects
        image_relevance = kitchen_annotations.groupby('ImageID').agg({
            'LabelName': lambda x: len(set(x)),  # Number of different relevant classes
            'Confidence': 'mean'  # Average confidence
        }).reset_index()
        
        # Sort by relevance (more diverse objects = more interesting)
        image_relevance = image_relevance.sort_values(
            ['LabelName', 'Confidence'], 
            ascending=[False, False]
        )
        
        # Take top images
        selected_images = image_relevance.head(max_images)['ImageID'].tolist()
        
        # Get full annotation data for selected images
        filtered_annotations = kitchen_annotations[
            kitchen_annotations['ImageID'].isin(selected_images)
        ]
        
        # Save filtered annotations
        output_path = self.base_dir / f"filtered_{split}_annotations.csv"
        filtered_annotations.to_csv(output_path, index=False)
        
        logger.info(f"✓ Filtered to {len(selected_images)} images with kitchen/safety content")
        return selected_images, filtered_annotations

    def download_images(self, image_ids, split='train', max_workers=8):
        """Download filtered images"""
        logger.info(f"Downloading {len(image_ids)} {split} images...")
        
        # Create split directory
        split_dir = self.base_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Load image metadata to get URLs
        images_df = pd.read_csv(self.base_dir / f"{split}_images.csv")
        selected_metadata = images_df[images_df['ImageID'].isin(image_ids)]
        
        def download_single_image(row):
            image_id = row['ImageID']
            image_url = row['OriginalURL']
            image_path = split_dir / f"{image_id}.jpg"
            
            if image_path.exists():
                return f"✓ {image_id} already exists"
            
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    return f"✓ Downloaded {image_id}"
                else:
                    return f"✗ Failed to download {image_id}: HTTP {response.status_code}"
            except Exception as e:
                return f"✗ Error downloading {image_id}: {str(e)}"
        
        # Download images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(download_single_image, selected_metadata.iterrows()))
        
        successful_downloads = sum(1 for r in results if r.startswith("✓ Downloaded"))
        logger.info(f"✓ Successfully downloaded {successful_downloads} new images")

    def create_yolo_annotations(self, split='train'):
        """Convert Open Images annotations to YOLO format"""
        logger.info(f"Converting {split} annotations to YOLO format...")
        
        # Load filtered annotations
        annotations_df = pd.read_csv(self.base_dir / f"filtered_{split}_annotations.csv")
        
        # Load images metadata for dimensions
        images_df = pd.read_csv(self.base_dir / f"{split}_images.csv")
        image_dims = dict(zip(images_df['ImageID'], 
                             zip(images_df['OriginalWidth'], images_df['OriginalHeight'])))
        
        # Create class mapping (target classes to indices)
        class_to_idx = {class_id: idx for idx, class_id in enumerate(self.target_classes.keys())}
        
        # Create labels directory
        labels_dir = self.base_dir / split / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        # Process each image
        for image_id, group in annotations_df.groupby('ImageID'):
            if image_id not in image_dims:
                continue
                
            img_width, img_height = image_dims[image_id]
            label_path = labels_dir / f"{image_id}.txt"
            
            with open(label_path, 'w') as f:
                for _, row in group.iterrows():
                    class_idx = class_to_idx[row['LabelName']]
                    
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (row['XMin'] + row['XMax']) / 2
                    y_center = (row['YMin'] + row['YMax']) / 2
                    width = row['XMax'] - row['XMin']
                    height = row['YMax'] - row['YMin']
                    
                    # Write YOLO annotation line
                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Save class names file
        classes_path = self.base_dir / "classes.txt"
        with open(classes_path, 'w') as f:
            for class_name in self.target_classes.values():
                f.write(f"{class_name}\n")
        
        logger.info(f"✓ Created YOLO annotations for {split}")

    def create_safety_scenarios(self, split='train'):
        """Create safety-specific annotations based on object combinations"""
        logger.info(f"Creating safety scenario annotations for {split}...")
        
        annotations_df = pd.read_csv(self.base_dir / f"filtered_{split}_annotations.csv")
        
        safety_scenarios = []
        
        for image_id, group in annotations_df.groupby('ImageID'):
            present_classes = set(group['LabelName'].tolist())
            present_class_names = [self.target_classes[cls] for cls in present_classes 
                                 if cls in self.target_classes]
            
            # Check for safety scenarios
            scenarios = []
            for scenario_name, required_classes in self.safety_labels.items():
                if any(req_class in present_class_names for req_class in required_classes):
                    scenarios.append(scenario_name)
            
            if scenarios:
                safety_scenarios.append({
                    'ImageID': image_id,
                    'safety_scenarios': scenarios,
                    'present_objects': present_class_names
                })
        
        # Save safety scenarios
        safety_df = pd.DataFrame(safety_scenarios)
        safety_df.to_csv(self.base_dir / f"safety_scenarios_{split}.csv", index=False)
        
        logger.info(f"✓ Found {len(safety_scenarios)} images with safety scenarios")
        return safety_scenarios

    def generate_dataset_summary(self):
        """Generate summary statistics of the prepared dataset"""
        logger.info("Generating dataset summary...")
        
        summary = {
            'total_target_classes': len(self.target_classes),
            'safety_scenarios': list(self.safety_labels.keys()),
            'splits': {}
        }
        
        for split in ['train', 'validation', 'test']:
            annotations_file = self.base_dir / f"filtered_{split}_annotations.csv"
            if annotations_file.exists():
                df = pd.read_csv(annotations_file)
                summary['splits'][split] = {
                    'total_images': df['ImageID'].nunique(),
                    'total_annotations': len(df),
                    'class_distribution': df['LabelName'].value_counts().to_dict()
                }
        
        # Save summary
        with open(self.base_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("✓ Dataset summary saved")
        return summary


def main():
    """Complete pipeline to prepare Open Images kitchen safety dataset"""
    
    # Initialize dataset handler
    dataset = OpenImagesKitchenSafetyDataset(base_dir="./openimages_kitchen_safety")
    
    try:
        # Step 1: Download metadata
        dataset.download_metadata()
        
        # Step 2: Filter images for each split
        splits_config = {
            'train': 10000,      # 10K training images
            'validation': 2000,   # 2K validation images  
            'test': 1000         # 1K test images
        }
        
        for split, max_images in splits_config.items():
            logger.info(f"\n=== Processing {split} split ===")
            
            # Filter relevant images
            image_ids, annotations = dataset.filter_kitchen_images(split, max_images)
            
            # Download images
            dataset.download_images(image_ids, split)
            
            # Create YOLO annotations
            dataset.create_yolo_annotations(split)
            
            # Create safety scenarios
            dataset.create_safety_scenarios(split)
        
        # Step 3: Generate summary
        summary = dataset.generate_dataset_summary()
        print("\n=== Dataset Summary ===")
        print(json.dumps(summary, indent=2))
        
        logger.info("\n🎉 Dataset preparation complete!")
        logger.info(f"Dataset location: {dataset.base_dir}")
        
    except Exception as e:
        logger.error(f"Error in dataset preparation: {e}")
        raise

if __name__ == "__main__":
    main()
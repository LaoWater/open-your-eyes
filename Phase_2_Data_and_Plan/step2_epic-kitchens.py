# Epic-Kitchens Dataset Kitchen Safety Setup
# Process video data for hand-object interaction safety detection

import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import json
import requests
import zipfile
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EpicKitchensKitchenSafety:
    def __init__(self, base_dir="./epic_kitchens_safety"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Safety-relevant action classes from Epic-Kitchens
        self.danger_actions = {
            'cut': ['knife_handling', 'sharp_object_risk'],
            'chop': ['knife_handling', 'sharp_object_risk'], 
            'slice': ['knife_handling', 'sharp_object_risk'],
            'pour': ['spill_risk', 'liquid_handling'],
            'fry': ['hot_surface_contact', 'oil_splatter_risk'],
            'boil': ['hot_surface_contact', 'steam_risk'],
            'wash': ['wet_surface_risk', 'slip_hazard'],
            'clean': ['chemical_exposure', 'wet_surface_risk'],
            'open': ['container_safety', 'potential_spill'],
            'close': ['container_safety', 'finger_pinch_risk']
        }
        
        # Objects that create safety contexts
        self.safety_objects = {
            'knife': 'sharp_object',
            'pan': 'hot_surface', 
            'pot': 'hot_surface',
            'stove': 'hot_surface',
            'oil': 'hot_liquid',
            'water': 'liquid_spill',
            'glass': 'fragile_object',
            'plate': 'fragile_object'
        }

    def download_epic_kitchens_data(self):
        """Download Epic-Kitchens annotations and setup"""
        logger.info("Setting up Epic-Kitchens dataset...")
        
        # Epic-Kitchens annotation URLs
        annotation_urls = {
            'train_actions': 'https://github.com/epic-kitchens/epic-kitchens-100-annotations/raw/master/EPIC_100_train.csv',
            'validation_actions': 'https://github.com/epic-kitchens/epic-kitchens-100-annotations/raw/master/EPIC_100_validation.csv',
            'noun_classes': 'https://github.com/epic-kitchens/epic-kitchens-100-annotations/raw/master/EPIC_100_noun_classes.csv',
            'verb_classes': 'https://github.com/epic-kitchens/epic-kitchens-100-annotations/raw/master/EPIC_100_verb_classes.csv'
        }
        
        for name, url in annotation_urls.items():
            filepath = self.base_dir / f"{name}.csv"
            if not filepath.exists():
                logger.info(f"Downloading {name}...")
                response = requests.get(url)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.info(f"✓ Downloaded {name}")

    def identify_safety_relevant_segments(self, split='train', min_confidence=0.8):
        """Identify video segments with safety-relevant actions"""
        logger.info(f"Identifying safety-relevant segments in {split}...")
        
        # Load annotations
        actions_df = pd.read_csv(self.base_dir / f"{split}_actions.csv")
        verbs_df = pd.read_csv(self.base_dir / "verb_classes.csv")
        nouns_df = pd.read_csv(self.base_dir / "noun_classes.csv")
        
        # Create verb and noun mappings
        verb_map = dict(zip(verbs_df['id'], verbs_df['key']))
        noun_map = dict(zip(nouns_df['id'], nouns_df['key']))
        
        # Add text labels to actions
        actions_df['verb_text'] = actions_df['verb'].map(verb_map)
        actions_df['noun_text'] = actions_df['noun'].map(noun_map)
        
        safety_segments = []
        
        for _, row in actions_df.iterrows():
            verb = row['verb_text']
            noun = row['noun_text']
            
            # Check if this action involves safety risks
            safety_risks = []
            
            # Check verb-based risks
            for danger_verb, risks in self.danger_actions.items():
                if danger_verb in verb.lower():
                    safety_risks.extend(risks)
            
            # Check noun-based risks  
            for safety_object, risk_type in self.safety_objects.items():
                if safety_object in noun.lower():
                    safety_risks.append(risk_type)
            
            if safety_risks:
                safety_segments.append({
                    'segment_id': row['uid'],
                    'video_id': row['video_id'],
                    'participant_id': row['participant_id'],
                    'start_timestamp': row['start_timestamp'],
                    'stop_timestamp': row['stop_timestamp'], 
                    'start_frame': row['start_frame'],
                    'stop_frame': row['stop_frame'],
                    'verb': verb,
                    'noun': noun,
                    'safety_risks': safety_risks,
                    'action_description': f"{verb} {noun}"
                })
        
        # Save safety segments
        safety_df = pd.DataFrame(safety_segments)
        safety_df.to_csv(self.base_dir / f"safety_segments_{split}.csv", index=False)
        
        logger.info(f"✓ Found {len(safety_segments)} safety-relevant segments")
        return safety_segments

    def extract_safety_frames(self, safety_segments, video_base_path, max_segments=1000):
        """Extract frames from safety-relevant video segments"""
        logger.info("Extracting frames from safety segments...")
        
        frames_dir = self.base_dir / "safety_frames"
        frames_dir.mkdir(exist_ok=True)
        
        extracted_frames = []
        
        # Process limited number of segments
        for i, segment in enumerate(safety_segments[:max_segments]):
            if i % 100 == 0:
                logger.info(f"Processing segment {i}/{min(max_segments, len(safety_segments))}")
            
            video_path = Path(video_base_path) / f"{segment['video_id']}.mp4"
            
            if not video_path.exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            try:
                # Open video
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Calculate frame numbers
                start_frame = int(segment['start_frame'])
                stop_frame = int(segment['stop_frame'])
                
                # Extract key frames (beginning, middle, end of action)
                key_frames = [
                    start_frame,
                    (start_frame + stop_frame) // 2,  # middle
                    stop_frame
                ]
                
                for frame_num in key_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Save frame
                        frame_filename = f"{segment['segment_id']}_frame_{frame_num}.jpg"
                        frame_path = frames_dir / frame_filename
                        cv2.imwrite(str(frame_path), frame)
                        
                        extracted_frames.append({
                            'frame_path': frame_filename,
                            'segment_id': segment['segment_id'],
                            'video_id': segment['video_id'],
                            'frame_number': frame_num,
                            'timestamp': frame_num / fps,
                            'safety_risks': segment['safety_risks'],
                            'action_description': segment['action_description'],
                            'verb': segment['verb'],
                            'noun': segment['noun']
                        })
                
                cap.release()
                
            except Exception as e:
                logger.error(f"Error processing segment {segment['segment_id']}: {e}")
        
        # Save frame metadata
        frames_df = pd.DataFrame(extracted_frames)
        frames_df.to_csv(self.base_dir / "extracted_frames_metadata.csv", index=False)
        
        logger.info(f"✓ Extracted {len(extracted_frames)} safety-relevant frames")
        return extracted_frames

    def create_safety_classification_dataset(self):
        """Create binary classification dataset for safety scenarios"""
        logger.info("Creating safety classification dataset...")
        
        frames_df = pd.read_csv(self.base_dir / "extracted_frames_metadata.csv")
        
        # Create safety labels
        safety_labels = []
        
        for _, frame in frames_df.iterrows():
            risks = eval(frame['safety_risks']) if isinstance(frame['safety_risks'], str) else frame['safety_risks']
            
            # Binary classification for each safety type
            labels = {
                'frame_path': frame['frame_path'],
                'knife_handling_risk': int('sharp_object_risk' in risks or 'knife_handling' in risks),
                'hot_surface_risk': int('hot_surface_contact' in risks or 'hot_surface' in risks),
                'spill_risk': int('spill_risk' in risks or 'liquid_handling' in risks),
                'slip_hazard': int('wet_surface_risk' in risks or 'slip_hazard' in risks),
                'general_safety_risk': int(len(risks) > 0),
                'action_verb': frame['verb'],
                'action_noun': frame['noun']
            }
            
            safety_labels.append(labels)
        
        # Save classification dataset
        classification_df = pd.DataFrame(safety_labels)
        classification_df.to_csv(self.base_dir / "safety_classification_labels.csv", index=False)
        
        # Create train/val split
        train_split = classification_df.sample(frac=0.8, random_state=42)
        val_split = classification_df.drop(train_split.index)
        
        train_split.to_csv(self.base_dir / "train_safety_labels.csv", index=False)
        val_split.to_csv(self.base_dir / "val_safety_labels.csv", index=False)
        
        logger.info(f"✓ Created classification dataset: {len(train_split)} train, {len(val_split)} val")
        
        return classification_df

    def create_yolo_annotations_from_epic(self):
        """Create YOLO-style annotations focusing on hand-object interactions"""
        logger.info("Creating YOLO annotations from Epic-Kitchens data...")
        
        # This would require additional object detection annotations
        # Epic-Kitchens provides action labels but not bounding boxes
        # You would need to:
        # 1. Run a pre-trained hand detector (like MediaPipe) on extracted frames
        # 2. Run object detection to find kitchen tools
        # 3. Create pseudo-labels based on temporal action annotations
        
        logger.info("Note: YOLO annotation creation requires additional object detection step")
        logger.info("Consider using MediaPipe Hands + YOLO object detection for pseudo-labeling")

    def generate_epic_summary(self):
        """Generate summary of Epic-Kitchens safety dataset"""
        summary = {}
        
        # Count safety segments
        for split in ['train', 'validation']:
            csv_path = self.base_dir / f"safety_segments_{split}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                summary[f'{split}_segments'] = len(df)
                summary[f'{split}_risk_distribution'] = {}
                
                # Count risk types
                all_risks = []
                for risks_str in df['safety_risks']:
                    risks = eval(risks_str) if isinstance(risks_str, str) else risks_str
                    all_risks.extend(risks)
                
                from collections import Counter
                risk_counts = Counter(all_risks)
                summary[f'{split}_risk_distribution'] = dict(risk_counts)
        
        # Save summary
        with open(self.base_dir / "epic_kitchens_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main_epic_kitchens():
    """Main pipeline for Epic-Kitchens safety dataset preparation"""
    
    dataset = EpicKitchensKitchenSafety()
    
    try:
        # Step 1: Download annotations
        dataset.download_epic_kitchens_data()
        
        # Step 2: Identify safety segments
        train_segments = dataset.identify_safety_relevant_segments('train')
        val_segments = dataset.identify_safety_relevant_segments('validation') 
        
        # Step 3: Extract frames (requires video files)
        # video_path = input("Enter path to Epic-Kitchens video files: ")
        # dataset.extract_safety_frames(train_segments, video_path)
        
        # Step 4: Create classification dataset
        # dataset.create_safety_classification_dataset()
        
        # Step 5: Generate summary
        summary = dataset.generate_epic_summary()
        print(json.dumps(summary, indent=2))
        
        logger.info("✓ Epic-Kitchens safety dataset preparation complete!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main_epic_kitchens()
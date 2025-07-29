# Script for preprocessing our images
import os
import yaml

import torch
import torchvision.transforms as transforms
from PIL import Image
import mlflow
import mlflow.pytorch

import logging
from tqdm import tqdm


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# To load params from params.yaml
def load_config(config_path: str = "params.yml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_transforms(config: dict, split: str) -> transforms.Compose:
    """Get transforms based on configuration and data split"""
    transform_config = config['preprocessing']['transforms']
    
    # Base transforms
    base_transforms = [
        transforms.Resize((transform_config['resize']['height'], 
                          transform_config['resize']['width'])),
    ]
    
    # Add augmentations only for the training data
    if split == 'train' and transform_config.get('augmentation', {}).get('enabled', False):
        aug_config = transform_config['augmentation']
        
        if aug_config.get('random_horizontal_flip', False):
            base_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
            
    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=transform_config['normalize']['mean'],
            std=transform_config['normalize']['std']
        )
    ])
    
    return transforms.Compose(base_transforms)

def process_images(input_dir: str, output_dir: str, transforms_fn: transforms.Compose, 
                  split: str, class_name: str) -> dict:
    """Process images from input directory and save to output directory."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    processed_count = 0
    failed_count = 0
    
    logger.info("Processing %d images from %s", len(image_files), input_dir)
    
    for filename in tqdm(image_files, desc=f"Processing {split}/{class_name}"):
        try:
            # Load image
            input_path = os.path.join(input_dir, filename)
            image = Image.open(input_path).convert('L') # Convert to grayscale
            
            # Apply transforms
            processed_tensor = transforms_fn(image)
            
            # Save processed tensor
            output_filename = f"{os.path.splitext(filename)[0]}.pt"
            output_path = os.path.join(output_dir, output_filename)
            torch.save(processed_tensor, output_path)
            
            processed_count += 1
            
        except Exception as e:
            logger.error("Failed to process %s: %s", filename, str(e))
            failed_count += 1
    
    return {
        'processed': processed_count,
        'failed': failed_count,
        'total': len(image_files)
    }

def main():
    """Main preprocessing function."""
    
    # Load configuration
    config = load_config()
    
    # Start MLflow run
    with mlflow.start_run(run_name="data_preprocessing"):
        
        # Log preprocessing parameters
        mlflow.log_params({
            "resize_height": config['preprocessing']['transforms']['resize']['height'],
            "resize_width": config['preprocessing']['transforms']['resize']['width'],
            "normalize_mean": config['preprocessing']['transforms']['normalize']['mean'],
            "normalize_std": config['preprocessing']['transforms']['normalize']['std'],
            "augmentation_enabled": config['preprocessing']['transforms']['augmentation']['enabled']
        })
        
        # Define paths
        raw_data_path = config['data']['raw_path']
        processed_data_path = config['data']['processed_path']
        
        splits = ['train', 'valid', 'test']
        classes = ['0', '1']  # benign, malignant
        
        total_stats = {'processed': 0, 'failed': 0, 'total': 0}
        
        for split in splits:
            logger.info("Processing %s split...", split)
            
            # Get transforms for this split
            transform_fn = get_transforms(config, split)
            
            for class_label in classes:
                input_dir = os.path.join(raw_data_path, split, class_label)
                output_dir = os.path.join(processed_data_path, split, class_label)
                
                if not os.path.exists(input_dir):
                    logger.warning("Input directory does not exist: %s", input_dir)
                    continue
                
                # Process images
                class_name = "benign" if class_label == "0" else "malignant"
                stats = process_images(input_dir, output_dir, transform_fn, split, class_name)
                
                # Update total stats
                for key in total_stats:
                    total_stats[key] += stats[key]
                
                # Log metrics for this class/split
                mlflow.log_metrics({
                    f"{split}_{class_name}_processed": stats['processed'],
                    f"{split}_{class_name}_failed": stats['failed'],
                    f"{split}_{class_name}_total": stats['total']
                })
                
                logger.info("Completed %s/%s: %d/%d processed, %d failed",
                            split, class_name, stats['processed'], stats['total'], stats['failed'])
        
        # Log overall metrics
        mlflow.log_metrics({
            "total_processed": total_stats['processed'],
            "total_failed": total_stats['failed'],
            "total_images": total_stats['total'],
            "success_rate": total_stats['processed'] / total_stats['total'] if total_stats['total'] > 0 else 0
        })
        
        logger.info("Preprocessing completed! Total: %d/%d processed", total_stats['processed'], total_stats['total'])
        # Log the preprocessing script as an artifact
        mlflow.log_artifact(__file__)

if __name__ == "__main__":
    main()

# Script for preprocessing our images
import os
import yaml

import torch
from torchvision import transforms
from PIL import Image
import mlflow

import logging
import traceback


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

def process_images(input_dir: str,
                  output_dir: str,
                  transform_fn: transforms.Compose,
                  sample_size: int=None) -> dict:
    """Process images from input directory and save to output directory."""
    
    # Check that the input dir exists
    if not os.path.exists(input_dir):
        logger.warning("input directory does not exist: %s", input_dir)
        return {'processed': 0, 'failed': 0, 'total': 0}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith('.jpg')]
    
    # Subsample data if specified
    if sample_size:
        image_files = image_files[:sample_size]

    processed_count = 0
    failed_count = 0
    
    logger.info("Processing %d images from %s", len(image_files), input_dir)
    
    for filename in image_files:
        try:
            # Load image
            input_path = os.path.join(input_dir, filename)
            logger.debug("Processing image: %s", input_path)
            image = Image.open(input_path).convert('L') # Convert to grayscale
            
            # Apply transforms
            processed_tensor = transform_fn(image)
            
            # Save processed tensor
            output_filename = f"{os.path.splitext(filename)[0]}.pt"
            output_path = os.path.join(output_dir, output_filename)
            torch.save(processed_tensor, output_path)
            
            logger.debug("Saved processed tensor to: %s", output_path)
            processed_count += 1
            
        except Exception as e:
            logger.error("Failed to process %s: %s", filename, str(e))
            logger.debug(traceback.format_exc())
            failed_count += 1
    
    return {
        'processed': processed_count,
        'failed': failed_count,
        'total': len(image_files)
    }

def run_preprocessing(config_path: str='params.yml',
                     testing_size: int=None, 
                     enable_mlflow: bool=True) -> dict:
    """Main preprocessing function."""
    
    # Load configuration
    config = load_config(config_path)
    logger.info("Loaded config from %s", config_path)
    logger.debug("Config contents: %s", config)
    
    # Define paths
    raw_data_path = config['data']['raw_path']
    processed_data_path = config['data']['processed_path']
    
    splits = ['train', 'valid', 'test']
    classes = ['0', '1']  # benign, malignant
    
    total_stats = {'processed': 0, 'failed': 0, 'total': 0}

    # Set mlflow run name
    run_name = "data_preprocessing"
    if testing_size:
        run_name = f"testing_{run_name}"

    # Star mlflow run if enabled
    mlflow_context = mlflow.start_run(run_name=run_name) if enable_mlflow else None
    try:
        if enable_mlflow:
            # Log preprocessing parameters
            mlflow.log_params({
                "resize_height": config['preprocessing']['transforms']['resize']['height'],
                "resize_width": config['preprocessing']['transforms']['resize']['width'],
                "normalize_mean": config['preprocessing']['transforms']['normalize']['mean'],
                "normalize_std": config['preprocessing']['transforms']['normalize']['std'],
                "augmentation_enabled": config['preprocessing']['transforms']['augmentation']['enabled'],
                'testing_size': testing_size
            })
        
        for split in splits:
            logger.info("Starting processing for %s split...", split)
            
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
                stats = process_images(input_dir, output_dir, transform_fn, testing_size)
                
                # Update total stats
                for key in total_stats:
                    total_stats[key] += stats[key]
                
                # Log metrics for this class/split
                if enable_mlflow:
                    mlflow.log_metrics({
                        f"{split}_{class_name}_processed": stats['processed'],
                        f"{split}_{class_name}_failed": stats['failed'],
                        f"{split}_{class_name}_total": stats['total']
                    })
                
                logger.info("Completed %s/%s: %d/%d processed, %d failed",
                           split, class_name, stats['processed'], stats['total'], stats['failed'])
            logger.info(f"Finished processing {split} split.")
        
        # Log overall metrics
        if enable_mlflow:
            mlflow.log_metrics({
                "total_processed": total_stats['processed'],
                "total_failed": total_stats['failed'],
                "total_images": total_stats['total'],
                "success_rate": total_stats['processed'] / total_stats['total'] if total_stats['total'] > 0 else 0
            })
        
        logger.info("Preprocessing completed! Total: %d/%d processed", total_stats['processed'], total_stats['total'])
        # Log the preprocessing script as an artifact
        if enable_mlflow:
            mlflow.log_artifact(__file__)

    finally:
        if mlflow_context:
            mlflow.end_run()

    return total_stats

if __name__ == "__main__":
    run_preprocessing()
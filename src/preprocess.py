# Script for preprocessing our images
from src.utils.secrets import load_credentials
import os
import yaml

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import mlflow

import logging
import traceback


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials to so we can set env variables
load_credentials()

# To load params from params.yaml
def load_config(config_path: str = "params.yaml") -> dict:
    """Load preprocessing configuration from YAML file"""
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


class BreastCancerDataset(Dataset):
    """Custom PT Datasaet to lad images and labels."""
    def __init__(self, images: list, labels: list):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def build_dataset(split_dir: str,
                  transform_fn: transforms.Compose,
                  classes: list,
                  sample_size: int = None) -> tuple:
    """Load images from class subdirs, apply transforms, return a Dataset object and stats of processed data.
    Args:
        split_dir (str): Path to dir containing class subdirectories for a given split.
        transform_fn (transforms.Compose): Transform pipeline to apply to each image.
        classes (list): List of class label names corresponding to subdirectory names.
        sample_size (int, optional): Maximum number of images to load per class. Defaults to None.
    Returns:
        tuple: A BreastCancerDataset instance and a stats dict with processed, failed, and total counts.
    """

    images = []
    labels = []
    stats = {'processed': 0, 'failed': 0, 'total': 0}

    for class_label in classes:
        class_dir = os.path.join(split_dir, class_label)

        # Handle missing class dir
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Class directory does not exist: {class_dir}")

        # Get list of image files
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
        # Subsample data for testing purposes
        if sample_size:
            image_files = image_files[:sample_size]

        stats['total'] += len(image_files)
        logger.info("Processing %d images from %s", len(image_files), class_dir)

        for filename in image_files:
            try:
                input_path = os.path.join(class_dir, filename)
                logger.debug("Processing image: %s", input_path)
                image = Image.open(input_path).convert('L') #Convert image to greyscale
                tensor = transform_fn(image)
                images.append(tensor)
                labels.append(int(class_label))
                stats['processed'] += 1
            except Exception as e:
                logger.error("Failed to process %s: %s", filename, str(e))
                logger.debug(traceback.format_exc())
                stats['failed'] += 1

    dataset = BreastCancerDataset(images, labels)
    return dataset, stats


def run_preprocessing(config_path: str = 'params.yaml',
                      testing_size: int = None,
                      enable_mlflow: bool = True) -> dict:
    """Main preprocessing function. Builds and saves one Dataset per split.
    Args:
        config_path (str): Path to the preprocessing config file. Defaults to 'params.yaml'.
        testing_size (int, optional): If provided, limits the number of samples per split (for testing purposes).
        enable_mlflow (bool): Whether to log parameters and metrics to MLflow. Defaults to True.
    Returns:
        dict: A dictionary containing preprocessing statistics for each split (e.g., loaded, failed counts).
    """
    
    config = load_config(config_path)
    logger.info("Loaded config from %s", config_path)
    logger.debug("Config contents: %s", config)

    raw_data_path = config['data']['raw_path']
    processed_data_path = config['data']['processed_path']

    os.makedirs(processed_data_path, exist_ok=True)

    splits = ['train', 'valid', 'test']
    classes = ['0', '1']  # benign, malignant
    total_stats = {'processed': 0, 'failed': 0, 'total': 0}

    # Config mlflow
    experiment_name = config.get('mlflow', {}).get('experiment_name', 'default')
    run_name = "data_preprocessing"
    if testing_size:
        run_name = f"testing_{run_name}"

    if enable_mlflow:
        uri = config.get('mlflow', {}).get('tracking_uri') or os.getenv('MLFLOW_TRACKING_URI')
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

    # Start mlflow run if enabled
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
            logger.info("Building dataset for %s split...", split)

            # Get transforms for this split
            transform_fn = get_transforms(config, split)
            split_dir = os.path.join(raw_data_path, split)

            # Build the dataset and compute statistics
            dataset, stats = build_dataset(split_dir, transform_fn, classes, testing_size)
            # Store datases
            output_path = os.path.join(processed_data_path, f"{split}_dataset.pt")
            torch.save(dataset, output_path)
            logger.info("Saved %s dataset (%d samples) to %s", split, len(dataset), output_path)

            # Log stats for this split
            for key in total_stats:
                total_stats[key] += stats[key]

            if enable_mlflow:
                mlflow.log_metrics({
                    f"{split}_processed": stats['processed'],
                    f"{split}_failed": stats['failed'],
                    f"{split}_total": stats['total']
                })

            logger.info("Completed %s split: %d/%d processed, %d failed",
                        split, stats['processed'], stats['total'], stats['failed'])

        # Log stats for all of our data
        if enable_mlflow:
            mlflow.log_metrics({
                "total_processed": total_stats['processed'],
                "total_failed": total_stats['failed'],
                "total_images": total_stats['total'],
                "success_rate": total_stats['processed'] / total_stats['total'] if total_stats['total'] > 0 else 0
            })

        logger.info("Preprocessing completed! Total: %d/%d processed",
                    total_stats['processed'], total_stats['total'])

        if enable_mlflow:
            mlflow.log_artifact(__file__)

    finally:
        if mlflow_context:
            mlflow.end_run()

    return total_stats


if __name__ == "__main__":
    run_preprocessing()

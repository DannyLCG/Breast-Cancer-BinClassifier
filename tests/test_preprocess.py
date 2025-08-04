# Script to test the preprocessing pipeline in src/preprocess.py using pytest
import os
import pytest
import yaml
import torch
from PIL import Image
import tempfile
import shutil
from pathlib import Path

# Import your preprocessing functions
import sys
sys.path.append('src')
from preprocess import run_preprocessing, get_transforms, process_images, load_config

@pytest.fixture
def mini_dataset(tmp_path):
    """Create a mini dataset with 3 images per split/class from existing data"""
    
    # Path to your actual data
    actual_raw_path = "data/raw"
    
    # Create temporary directory structure
    temp_raw_path = tmp_path / "data" / "raw"
    splits = ['train', 'valid', 'test']
    classes = ['0', '1']
    
    for split in splits:
        for class_label in classes:
            # Source directory (your actual data)
            source_dir = Path(actual_raw_path) / split / class_label
            
            # Destination directory (temporary test data)
            dest_dir = temp_raw_path / split / class_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            if source_dir.exists():
                # Get first 3 image files from your actual data
                image_files = [f for f in os.listdir(source_dir) 
                              if f.lower().endswith('.jpg')][:3]
                
                # Copy 3 images to temp directory
                for img_file in image_files:
                    src_path = source_dir / img_file
                    dst_path = dest_dir / img_file
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied {src_path} -> {dst_path}")
            else:
                print(f"Warning: Source directory {source_dir} does not exist")
    
    return tmp_path

@pytest.fixture
def test_config(mini_dataset):
    """Create a test config that uses your actual params.yml but with temp paths"""
    
    # Load your actual params.yml
    with open("params.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Override paths to use temporary directories
    config['data']['raw_path'] = str(mini_dataset / "data" / "raw")
    config['data']['processed_path'] = str(mini_dataset / "data" / "processed")
    
    # Save modified config to temp location
    config_path = mini_dataset / "test_params.yml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)

def test_load_config(test_config):
    """Test that config loading works"""
    config = load_config(test_config)
    assert config['data']['raw_path'] is not None
    assert config['preprocessing']['transforms']['resize']['height'] == 224
    assert config['preprocessing']['transforms']['resize']['width'] == 224

def test_get_transforms(test_config):
    """Test transform creation for different splits"""
    config = load_config(test_config)
    
    # Test train transforms (should include augmentation)
    train_transforms = get_transforms(config, 'train')
    assert train_transforms is not None
    
    # Test validation transforms (should not include augmentation)
    val_transforms = get_transforms(config, 'valid')
    assert val_transforms is not None

def test_process_images_single_class(mini_dataset, test_config):
    """Test processing images for a single class"""
    config = load_config(test_config)
    
    # Test processing train/benign images
    input_dir = os.path.join(config['data']['raw_path'], 'train', '0')
    output_dir = os.path.join(config['data']['processed_path'], 'train', '0')
    
    transform_fn = get_transforms(config, 'train')
    
    stats = process_images(
        input_dir=input_dir,
        output_dir=output_dir,
        transform_fn=transform_fn,
        sample_size=3
    )
    
    # Check stats
    assert stats['processed'] == 3
    assert stats['failed'] == 0
    assert stats['total'] == 3
    
    # Check that output files exist
    assert os.path.exists(output_dir)
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    assert len(output_files) == 3
    
    # Check that tensors can be loaded
    for file in output_files:
        tensor = torch.load(os.path.join(output_dir, file))
        assert tensor.shape == (1, 224, 224)  # Grayscale, 224x224

def test_full_preprocessing_pipeline(mini_dataset, test_config):
    """Test the complete preprocessing pipeline with your actual config"""
    
    print(f"\nUsing test config: {test_config}")
    print(f"Mini dataset location: {mini_dataset}")
    
    # Load config to check what we're testing with
    config = load_config(test_config)
    print(f"Image size: {config['preprocessing']['transforms']['resize']}")
    print(f"Normalization: mean={config['preprocessing']['transforms']['normalize']['mean']}, "
          f"std={config['preprocessing']['transforms']['normalize']['std']}")
    
    # Run preprocessing without MLflow (for testing)
    stats = run_preprocessing(
        config_path=test_config,
        testing_size=3,  # Process all 3 images per class/split
        enable_mlflow=False
    )
    
    print(f"\nProcessing results: {stats}")
    
    # Check overall stats
    assert stats['processed'] > 0, f"No images were processed. Stats: {stats}"
    assert stats['failed'] == 0, f"Some images failed to process. Stats: {stats}"
    
    # Check that processed directories exist and contain .pt files
    processed_path = config['data']['processed_path']
    splits = ['train', 'valid', 'test']
    classes = ['0', '1']
    
    total_pt_files = 0
    for split in splits:
        for class_label in classes:
            output_dir = os.path.join(processed_path, split, class_label)
            
            if os.path.exists(output_dir):
                pt_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
                total_pt_files += len(pt_files)
                print(f"{split}/{class_label}: {len(pt_files)} .pt files")
                
                # Test loading one tensor to verify it works
                if pt_files:
                    sample_tensor = torch.load(os.path.join(output_dir, pt_files[0]))
                    print(f"Sample tensor shape: {sample_tensor.shape}, dtype: {sample_tensor.dtype}")
                    
                    # Basic tensor checks
                    assert sample_tensor.dtype == torch.float32
                    assert len(sample_tensor.shape) == 3  # Should be (C, H, W)
    
    print(f"\nTotal .pt files created: {total_pt_files}")
    assert total_pt_files > 0, "No .pt files were created"
    
    print("\nTest passed! The preprocessing pipeline works correctly.")

def test_tensor_properties(mini_dataset, test_config):
    """Test that processed tensors have correct properties"""
    config = load_config(test_config)
    
    # Process one image
    input_dir = os.path.join(config['data']['raw_path'], 'test', '1')
    output_dir = os.path.join(config['data']['processed_path'], 'test', '1')
    
    transform_fn = get_transforms(config, 'test')
    
    process_images(
        input_dir=input_dir,
        output_dir=output_dir,
        transform_fn=transform_fn,
        sample_size=1
    )
    
    # Load the processed tensor
    pt_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    tensor = torch.load(os.path.join(output_dir, pt_files[0]))
    
    # Check tensor properties
    assert tensor.dtype == torch.float32
    assert tensor.shape == (1, 224, 224)  # Grayscale, resized to 224x224
    assert tensor.min() >= -3.0  # Reasonable range after normalization
    assert tensor.max() <= 3.0

def test_missing_input_directory(mini_dataset, test_config):
    """Test handling of missing input directories"""
    config = load_config(test_config)
    
    # Try to process from non-existent directory
    non_existent_dir = os.path.join(config['data']['raw_path'], 'nonexistent', '0')
    output_dir = os.path.join(config['data']['processed_path'], 'nonexistent', '0')
    
    transform_fn = get_transforms(config, 'test')
    
    # This should handle the missing directory gracefully
    # (Your current code logs a warning and continues)
    stats = process_images(
        input_dir=non_existent_dir,
        output_dir=output_dir,
        transform_fn=transform_fn,
        sample_size=1
    )
    
    # Should return zero stats for non-existent directory
    assert stats['processed'] == 0
    assert stats['total'] == 0
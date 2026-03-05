# Script to test the preprocessing pipeline in src/preprocess.py using pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml
import torch
import shutil

from torch.utils.data import DataLoader

import sys
sys.path.append('src')
from preprocess import run_preprocessing, get_transforms, build_dataset, load_config, BreastCancerDataset

@pytest.fixture
def mini_dataset(tmp_path):
    """Create a mini dataset with 3 images per split/class from existing data"""

    # Path to actual data
    actual_raw_path = "data/raw"

    # Create temporary directory structure
    temp_raw_path = tmp_path / "data" / "raw"
    splits = ['train', 'valid', 'test']
    classes = ['0', '1']

    for split in splits:
        for class_label in classes:
            # Source dir to actual data
            source_dir = Path(actual_raw_path) / split / class_label

            # Destination directory (temporary test data)
            dest_dir = temp_raw_path / split / class_label
            dest_dir.mkdir(parents=True, exist_ok=True)

            if source_dir.exists():
                # Get first 3 image files from our data
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
    """Create a test config copying params from the actual params.yaml file but with temp paths"""

    # Load actual params.yaml
    with open("params.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Override paths to use temporary directories
    config['data']['raw_path'] = str(mini_dataset / "data" / "raw")
    config['data']['processed_path'] = str(mini_dataset / "data" / "processed")

    # Save modified config to temp location
    config_path = mini_dataset / "test_params.yaml"
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

    train_transforms = get_transforms(config, 'train')
    assert train_transforms is not None

    val_transforms = get_transforms(config, 'valid')
    assert val_transforms is not None


def test_breast_cancer_dataset():
    """Test BreastCancerDataset behaves correctly"""
    images = [torch.zeros(1, 224, 224), torch.ones(1, 224, 224)]
    labels = [0, 1]

    dataset = BreastCancerDataset(images, labels)

    assert len(dataset) == 2
    img, label = dataset[0]
    assert img.shape == (1, 224, 224)
    assert label == 0


def test_build_dataset(mini_dataset, test_config):
    """Test build_dataset returns a BreastCancerDataset with correct structure"""
    config = load_config(test_config)
    transform_fn = get_transforms(config, 'train')
    split_dir = os.path.join(config['data']['raw_path'], 'train')
    classes = ['0', '1']

    dataset, stats = build_dataset(split_dir, transform_fn, classes, sample_size=3)

    assert isinstance(dataset, BreastCancerDataset)
    assert stats['processed'] > 0
    assert stats['failed'] == 0
    assert stats['processed'] == stats['total']

    img, label = dataset[0]
    assert img.dtype == torch.float32
    assert len(img.shape) == 3
    assert label in [0, 1]


def test_build_dataset_missing_class_dir(mini_dataset, test_config):
    """Test build_dataset handles a missing class directory gracefully"""
    config = load_config(test_config)
    transform_fn = get_transforms(config, 'train')
    split_dir = os.path.join(config['data']['raw_path'], 'nonexistent_split')
    classes = ['0', '1']

    dataset, stats = build_dataset(split_dir, transform_fn, classes, sample_size=3)

    assert isinstance(dataset, BreastCancerDataset)
    assert len(dataset) == 0
    assert stats['processed'] == 0


def test_full_preprocessing_pipeline(mini_dataset, test_config):
    """Test the complete preprocessing pipeline produces one .pt Dataset file per split"""

    print(f"\nUsing test config: {test_config}")
    print(f"Mini dataset location: {mini_dataset}")

    config = load_config(test_config)
    print(f"Image size: {config['preprocessing']['transforms']['resize']}")

    stats = run_preprocessing(
        config_path=test_config,
        testing_size=3,
        enable_mlflow=False
    )

    print(f"\nProcessing results: {stats}")

    assert stats['processed'] > 0, f"No images were processed. Stats: {stats}"
    assert stats['failed'] == 0, f"Some images failed to process. Stats: {stats}"

    processed_path = config['data']['processed_path']
    splits = ['train', 'valid', 'test']

    for split in splits:
        dataset_path = os.path.join(processed_path, f"{split}_dataset.pt")
        assert os.path.exists(dataset_path), f"Missing dataset file: {dataset_path}"

        dataset = torch.load(dataset_path, weights_only=False)
        assert isinstance(dataset, BreastCancerDataset), \
            f"Expected BreastCancerDataset, got {type(dataset)}"
        assert len(dataset) > 0, f"{split} dataset is empty"

        img, label = dataset[0]
        assert img.dtype == torch.float32
        assert len(img.shape) == 3
        assert label in [0, 1]
        print(f"{split}: {len(dataset)} samples, tensor shape: {img.shape}")

    print("\nTest passed! The preprocessing pipeline works correctly.")


def test_dataset_dataloader_compatible(mini_dataset, test_config):
    """Test that the saved Dataset can be used with a DataLoader"""
    config = load_config(test_config)

    run_preprocessing(
        config_path=test_config,
        testing_size=3,
        enable_mlflow=False
    )

    processed_path = config['data']['processed_path']
    dataset_path = os.path.join(processed_path, "train_dataset.pt")

    dataset = torch.load(dataset_path, weights_only=False)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch_imgs, batch_labels = next(iter(loader))
    assert batch_imgs.ndim == 4
    assert batch_imgs.shape[1:] == torch.Size([1, 224, 224])
    assert batch_labels.ndim == 1


def test_tensor_properties(mini_dataset, test_config):
    """Test that tensors stored in the Dataset have correct properties after normalization"""
    config = load_config(test_config)
    transform_fn = get_transforms(config, 'test')
    split_dir = os.path.join(config['data']['raw_path'], 'test')
    classes = ['0', '1']

    dataset, _ = build_dataset(split_dir, transform_fn, classes, sample_size=1)

    assert len(dataset) > 0

    for i in range(len(dataset)):
        tensor, label = dataset[i]
        assert tensor.dtype == torch.float32
        assert tensor.shape == (1, 224, 224)
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0


@patch('preprocess.mlflow')
def test_mlflow_uri_from_env_var(mock_mlflow, mini_dataset, test_config):
    """Test that MLFLOW_TRACKING_URI env var is used when not set in config"""
    config = load_config(test_config)
    del config['mlflow']['tracking_uri']
    config_path = mini_dataset / "test_params_no_uri.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    env_vars = {
        'MLFLOW_TRACKING_URI': 'http://test-mlflow-host:5000',
        'MLFLOW_TRACKING_USERNAME': 'test-user',
        'MLFLOW_TRACKING_PASSWORD': 'test-pass',
    }
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict(os.environ, env_vars):
        run_preprocessing(config_path=str(config_path), testing_size=3, enable_mlflow=True)

    mock_mlflow.set_tracking_uri.assert_called_once_with('http://test-mlflow-host:5000')
    mock_mlflow.set_experiment.assert_called_once()


@patch('preprocess.mlflow')
def test_mlflow_uri_config_takes_precedence(mock_mlflow, mini_dataset, test_config):
    """Test that config tracking_uri takes precedence over the env var"""
    config = load_config(test_config)
    config['mlflow']['tracking_uri'] = 'http://config-mlflow-host:5000'
    config_path = mini_dataset / "test_params_with_uri.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    env_vars = {
        'MLFLOW_TRACKING_URI': 'http://env-mlflow-host:5000',
        'MLFLOW_TRACKING_USERNAME': 'test-user',
        'MLFLOW_TRACKING_PASSWORD': 'test-pass',
    }
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict(os.environ, env_vars):
        run_preprocessing(config_path=str(config_path), testing_size=3, enable_mlflow=True)

    mock_mlflow.set_tracking_uri.assert_called_once_with('http://config-mlflow-host:5000')


@patch('preprocess.mlflow')
def test_mlflow_credentials_available_in_env(mock_mlflow, mini_dataset, test_config):
    """Test that USERNAME and PASSWORD from .env are present in os.environ during preprocessing"""
    env_vars = {
        'MLFLOW_TRACKING_URI': 'http://test-mlflow-host:5000',
        'MLFLOW_TRACKING_USERNAME': 'test-user',
        'MLFLOW_TRACKING_PASSWORD': 'test-pass',
    }
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict(os.environ, env_vars):
        run_preprocessing(config_path=test_config, testing_size=3, enable_mlflow=True)
        assert os.environ.get('MLFLOW_TRACKING_USERNAME') == 'test-user'
        assert os.environ.get('MLFLOW_TRACKING_PASSWORD') == 'test-pass'

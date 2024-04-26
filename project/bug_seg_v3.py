import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, ResizeWithPadOrCropd,
    ScaleIntensityRanged, RandRotate90d, RandFlipd, Transform,
    AsDiscrete, Lambda, MapTransform, ToTensor
)
from monai.data import CacheDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import glob
import os
from torch.optim import Adam
from skimage import filters
import tifffile as tiff


# Custom transformation for loading images
class ImageioLoadImage(Transform):
    def __call__(self, data):
        loaded_data = {}
        for key, path in data.items():
            image = imageio.imread(path)
            loaded_data[key] = image
        return loaded_data


# Data loading function
def get_data(dataset_path: Path):
    dataset = []
    all_tif_files = glob.glob(os.path.join(dataset_path, '*.tif'))
    all_tif_files = [f for f in all_tif_files if '_sg.tif' not in f]
    for file_path in all_tif_files:
        base, ext = os.path.splitext(file_path)
        label_path = f"{base}_sg{ext}"
        if os.path.exists(label_path):
            dataset.append({'image': file_path, 'label': label_path})
    return dataset


def calculate_accuracy(y_pred, y_true):
    # y_pred and y_true are PyTorch tensors of shape (batch_size, channels, height, width)
    # and that y_pred has been passed through a softmax or similar function if necessary
    y_pred = torch.argmax(y_pred, dim=1)  # Convert probabilities to predicted class by taking the argmax
    correct = (y_pred == y_true).float()  # Convert boolean values to floats for summation
    accuracy = correct.sum() / correct.numel()  # Calculate accuracy
    return accuracy


# Load and preprocess labels
class PreprocessLabels(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] > 0).astype(np.uint8)
        return d


# Setting up data and transformations
base_directory = 'data/64/ACC'
dataset = get_data(Path(base_directory))
trainval_dicts, test_dicts = train_test_split(dataset, test_size=0.15, random_state=42)
train_dicts, val_dicts = train_test_split(trainval_dicts, test_size=0.176, random_state=42)

train_transforms = Compose([
    ImageioLoadImage(),
    EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
    ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=[128, 128, 128]),
    ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=3),
    RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
    PreprocessLabels(keys=['label']),
    ToTensor()
])

val_transforms = Compose([
    ImageioLoadImage(),
    EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
    ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=[128, 128, 128]),
    ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
    PreprocessLabels(keys=['label']),
    ToTensor()
])

test_transforms = Compose([
    ImageioLoadImage(),
    EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
    ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=[128, 128, 128]),
    ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
    PreprocessLabels(keys=['label']),
    ToTensor()
])

# Create datasets and data loaders
train_dataset = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=1.0, num_workers=4)
val_dataset = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=1.0, num_workers=4)
test_dataset = CacheDataset(data=test_dicts, transform=test_transforms, cache_rate=1.0, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Initialize the model and set up loss and optimizer
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout=0.2
)

loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = Adam(model.parameters(), lr=1e-3)
inference = SlidingWindowInferer(roi_size=(320, 320, -1), sw_batch_size=1)

# Containers for losses and scores
training_losses = []
validation_losses = []
validation_scores = []

# Training and validation loops
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for tr_data in train_loader:
        inputs, targets = tr_data['image'], tr_data['label']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    training_losses.append(epoch_loss / len(train_loader))

    model.eval()
    val_loss = 0
    val_dice_metric = DiceMetric(include_background=False, reduction="mean")
    with torch.no_grad():
        for val_data in val_loader:
            inputs, targets = val_data['image'], val_data['label']
            predictions = inference(inputs=inputs, network=model)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()
            val_dice_metric(y_pred=predictions, y=targets)

    validation_losses.append(val_loss / len(val_loader))
    validation_scores.append(val_dice_metric.aggregate().item())
    val_dice_metric.reset()

    print(
        f'Epoch {epoch + 1}, Training Loss: {training_losses[-1]:.4f}, Validation Loss: {validation_losses[-1]:.4f}, Dice Score: {validation_scores[-1]:.4f}')

    if (epoch + 1) % 5 == 0:  # Adjust this condition to change frequency
        with torch.no_grad():
            sample_data = next(iter(val_loader))
            sample_inputs, sample_labels = sample_data['image'], sample_data['label']
            sample_outputs = model(sample_inputs)
            sample_output = torch.argmax(sample_outputs, dim=1).unsqueeze(
                1).float()  # Ensure it has a channel dimension

            # Select a middle slice from the volume
            slice_idx = sample_inputs.shape[2] // 2  # Assuming the depth is the third dimension

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.title("Input Image Slice")
            plt.imshow(sample_inputs[0, 0, slice_idx].cpu(), cmap='gray')  # Adjust indexing if needed
            plt.subplot(1, 3, 2)
            plt.title("True Mask Slice")
            plt.imshow(sample_labels[0, 0, slice_idx].cpu(), cmap='gray')  # Adjust indexing if needed
            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask Slice")
            plt.imshow(sample_output[0, 0, slice_idx].cpu(), cmap='gray')  # Adjust indexing if needed
            plt.show()

# Plot training and validation losses and Dice scores
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), training_losses, label='Training Loss')
plt.plot(range(1, NUM_EPOCHS + 1), validation_losses, label='Validation Loss')
plt.plot(range(1, NUM_EPOCHS + 1), validation_scores, label='Dice Score')
plt.xlabel('Epochs')
plt.ylabel('Loss / Dice Score')
plt.title('Training and Validation Losses and Dice Scores')
plt.legend()
plt.show()

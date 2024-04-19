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
    y_pred = torch.argmax(y_pred, dim=1)
    correct = (y_pred == y_true).float()
    return correct.sum() / correct.numel()


class PreprocessLabels(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] > 0).astype(np.uint8)
        return d


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

train_dataset = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=1.0, num_workers=4)
val_dataset = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=1.0, num_workers=4)
test_dataset = CacheDataset(data=test_dicts, transform=val_transforms, cache_rate=1.0, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

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

training_losses = []
validation_losses = []
validation_scores = []
best_dice_score = 0.0

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
    processed_batches = 0

    with torch.no_grad():
        for val_data in val_loader:
            inputs, targets = val_data['image'], val_data['label']
            predictions = inference(inputs=inputs, network=model)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()
            val_dice_metric(y_pred=predictions, y=targets)
            processed_batches += 1

    if processed_batches > 0:
        average_val_loss = val_loss / processed_batches
        validation_losses.append(average_val_loss)
        current_dice_score = val_dice_metric.aggregate().item()
        validation_scores.append(current_dice_score)  # Ensure it's updated every epoch

        if current_dice_score > best_dice_score:
            best_dice_score = current_dice_score
            torch.save(model.state_dict(), 'model_best_checkpoint.pth')
            print(f"New best model saved at epoch {epoch} with Dice Score: {current_dice_score:.4f}")

        print(
            f'Epoch {epoch + 1}, Training Loss: {training_losses[-1]:.4f}, Validation Loss: {average_val_loss:.4f}, Dice Score: {current_dice_score:.4f}')
    else:
        print(f"No validation data processed for epoch {epoch + 1}")

if validation_scores:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), training_losses, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), validation_losses, label='Validation Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), validation_scores, label='Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Dice Score')
    plt.title('Training and Validation Losses and Dice Scores')
    plt.legend()
    plt.show()
else:
    print("No validation scores to plot.")

model.eval()
test_loss = 0
test_dice_metric = DiceMetric(include_background=False, reduction="mean")
with torch.no_grad():
    for batch_data in test_loader:
        inputs, targets = batch_data['image'], batch_data['label']
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        test_loss += loss.item()
        test_dice_metric(y_pred=outputs, y=targets)
average_dice = test_dice_metric.aggregate().item()
print(f"Average test loss: {test_loss / len(test_loader)}")
print(f"Average Dice Score: {average_dice:.4f}")

model.eval()
total_accuracy = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data['image'], data['label']
        outputs = model(inputs)
        accuracy = calculate_accuracy(outputs, labels)
        total_accuracy += accuracy.item()

average_accuracy = total_accuracy / len(test_loader)
print(f'Average Accuracy: {average_accuracy:.4f}')

import os
import glob
from pathlib import Path

from monai.data import Dataset, DataLoader, CacheDataset
import tifffile as tiff
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ResizeWithPadOrCropd, \
    ScaleIntensityRanged, RandRotate90d, RandFlipd


# Function to load a .tif image
def load_tif_image(path):
    return tiff.imread(path)


# Function to get dataset


# Dataset class for handling image loading and transformation
# class TiffDataset(Dataset):
#     def __init__(self, data, transform=None):
#         self.data = data
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#
#         # Load images using the custom function
#         image = load_tif_image(item['image'])
#         label = load_tif_image(item['label'])
#
#         # Prepare the dictionary to be processed by MONAI transforms
#         item = {'image': image, 'label': label}
#
#         # Apply MONAI transformations if they exist
#         if self.transform:
#             item = self.transform(item)
#
#         return item
#

# Setup dataset and dataloader

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


base_directory = 'data/64/ACC'
dataset = get_data(Path(base_directory))
trainval_dicts, test_dicts = train_test_split(dataset, test_size=0.15, random_state=42)
train_dicts, val_dicts = train_test_split(trainval_dicts, test_size=0.176, random_state=42)

# train_transforms = Compose([
#     EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
#     ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=[128, 128, 128]),
#     ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
#     RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=3),
#     RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
# ])
#
# train_dataset = CacheDataset(
#     data=train_dicts,
#     transform=train_transforms,
#     cache_rate=1.0,
#     num_workers=4
# )
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
# first_image = next(iter(train_loader))
# print("Size after changing its voxel resolution: ", first_image['image'].shape)

#
# print(len(train_dicts))
# print(len(val_dicts))
# print(len(test_dicts))
all_sizes = []
for i in dataset:
    all_sizes.append(load_tif_image(i['image']).shape)
print(all_sizes)
# train_ds = TiffDataset(data=train_dicts)
# val_ds = TiffDataset(data=val_dicts)
#
# train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
#

# tiff_img = load_tif_image(train_dicts[0]['image'])
# tiff_segmented_img = load_tif_image(train_dicts[0]['label'])
#
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(tiff_img[28], cmap='gray')
# ax[0].axis('off')  # Hide the axes
# ax[0].set_title("Original Image")
# ax[1].imshow(tiff_segmented_img[28], cmap='gray')
# ax[1].axis('off')
# ax[1].set_title("Segmented Image")
# plt.show()

# dummy_transforms = monai.transforms.Compose([
#     monai.transforms.LoadImaged(keys=['image', 'label']),
#     monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
# ])
#
# dummy_dataset = TiffDataset(data=train_dicts, transform=dummy_transforms)
#
# dummy_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=False)

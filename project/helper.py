import os
import glob
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage import filters


def find_and_process_series(base_directory):
    # Find all tif files in the base directory
    all_tif_files = glob.glob(os.path.join(base_directory, '*.tif'))

    # Process each file
    for file_path in all_tif_files:
        if '_sg.tif' in file_path:
            print(f"Skipped (already processed): {file_path}")
            continue
        # Load the image using tifffile
        img = tiff.imread(file_path)

        # Apply Otsu's threshold
        threshold_value = filters.threshold_otsu(img)
        binary_img = img > threshold_value  # Create a binary image

        # Generate the new path by inserting 'segmented_' before '.tif' in the filename
        base, ext = os.path.splitext(file_path)
        output_filepath = f"{base}_sg{ext}"

        # Check if the output file already exists
        if not os.path.exists(output_filepath):
            # Save the segmented image with the new name using tifffile
            tiff.imwrite(output_filepath, binary_img.astype(np.uint8) * 255)
            print(f"Saved: {output_filepath}")
        else:
            print(f"File already exists: {output_filepath}")


def get_data(dataset_path: Path):
    dataset = []

    # Find all tif files in the dataset_path
    all_tif_files = glob.glob(os.path.join(dataset_path, '*.tif'))

    # Filter out files that are already labeled with '_sg'
    all_tif_files = [f for f in all_tif_files if not '_sg.tif' in f]

    for file_path in all_tif_files:
        base, ext = os.path.splitext(file_path)
        label_path = f"{base}_sg{ext}"

        # Check if the labeled file exists
        if os.path.exists(label_path):
            data_entry = {
                'image': file_path,
                'label': label_path
            }
            dataset.append(data_entry)
        else:
            print(f"Segmented file does not exist for {file_path}")

    return dataset


# Specify the base directory where the files are located
base_directory = 'data/64/ACC'

# Start processing
# find_and_process_series(base_directory)
print(get_data(base_directory))

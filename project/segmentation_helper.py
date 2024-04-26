import os
import glob
import tifffile as tiff
import numpy as np
from skimage import filters


def find_and_process_series(base_directory):
    # Find all TIFF files in the base directory
    all_tif_files = glob.glob(os.path.join(base_directory, '*.tif'))

    # Define the subfolder for processed files
    processed_folder = os.path.join(base_directory, 'processed')

    # Create the 'processed' subfolder if it doesn't exist
    if not os.path.exists(processed_folder):
        os.mkdir(processed_folder)

    # Process each file
    for file_path in all_tif_files:
        if '_label.tif' in file_path:
            print(f"Skipped (already processed): {file_path}")
            continue

        # Load the image using tifffile
        img = tiff.imread(file_path)

        # Apply Otsu's threshold
        threshold_value = filters.threshold_otsu(img)
        binary_img = img > threshold_value  # Create a binary image

        # Generate the new path in the 'processed' folder
        base_name = os.path.basename(file_path)
        base, ext = os.path.splitext(base_name)
        output_filepath = os.path.join(processed_folder, f"{base}_label{ext}")

        # Check if the output file already exists
        if not os.path.exists(output_filepath):
            # Save the segmented image with the new name
            tiff.imwrite(output_filepath, binary_img.astype(np.uint8) * 255)
            print(f"Saved: {output_filepath}")
        else:
            print(f"File already exists: {output_filepath}")


# Specify the base directory where the files are located
base_directory = 'data/kappo/val'

# Start processing
find_and_process_series(base_directory)

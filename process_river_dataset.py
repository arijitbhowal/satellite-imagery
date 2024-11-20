
import os
import rasterio
import numpy as np
import cv2
import glob
import shutil

# Path to your dataset folder
train_images_folder = '/river_dataset'

# Look for directories ending with '_13'
folders = glob.glob(os.path.join(train_images_folder, '*_13*'))
print(f"Found {len(folders)} folders to process.")

processed_count = 0

# Process each folder
for index, folder in enumerate(folders):
    print(f"Processing {index+1}/{len(folders)} folder: {folder}")

    # Path to masks directory
    masks_dir = os.path.join(folder, 'masks')

    # Ensure masks directory exists
    if not os.path.exists(masks_dir):
        print(f"Skipping {folder}: Masks directory not found.")
        continue

    # Get all .tif mask files in the masks directory
    mask_files = glob.glob(os.path.join(masks_dir, '*.tif'))
    mask_files.sort()  # Sort files by name (assumes chronological naming)

    print(f"Mask files found: {mask_files}")

    # Check if there are at least two mask files
    if len(mask_files) < 2:
        print(f"Skipping {folder}: Not enough mask files.")
        continue

    # Select earliest and latest mask files
    earliest_mask = mask_files[0]
    latest_mask = mask_files[-1]

    # Path to the 'rivers' directory
    rivers_dir = os.path.join(folder, 'rivers')
    os.makedirs(rivers_dir, exist_ok=True)

    # Copy the masks as rivers1.tif and rivers2.tif
    r1_path = os.path.join(rivers_dir, 'rivers1.tif')
    r2_path = os.path.join(rivers_dir, 'rivers2.tif')
    shutil.copy(earliest_mask, r1_path)
    shutil.copy(latest_mask, r2_path)

    # Path to the 'change' directory
    change_dir = os.path.join(folder, 'change')
    if os.path.exists(change_dir):
        shutil.rmtree(change_dir)
    os.makedirs(change_dir, exist_ok=True)

    # Perform change detection
    with rasterio.open(r1_path) as src1, rasterio.open(r2_path) as src2:
        r1 = src1.read(1)
        r2 = src2.read(1)

    # Update r2 to zero out areas where r1 == 1
    r2[np.where(r1 == 1)] = 0

    # Save the change detection result
    change_output_path = os.path.join(change_dir, 'change.tif')
    cv2.imwrite(change_output_path, r2)

    processed_count += 1

print(f"Processing complete. Total folders processed: {processed_count}.")


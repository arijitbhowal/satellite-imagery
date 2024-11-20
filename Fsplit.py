
import numpy as np
import os
import glob
import random
import shutil

# Set the path directly
train_images_folder = '/river_dataset'

# Match all subfolders containing '_13'
FOLDER = glob.glob(os.path.join(train_images_folder, '*_13*'))

# Shuffle the folders
if len(FOLDER) >= 7:  # Ensure enough folders are present
    random.shuffle(FOLDER)
    Ftrain = FOLDER[:5]  # 5 folders for training
    Fval = FOLDER[5:6]   # 1 folder for validation
    Ftest = FOLDER[6:]       # All folders for testing
else:
    print(f"Not enough folders found! Only {len(FOLDER)} available.")
    Ftrain, Fval, Ftest = [], [], FOLDER

# Create output directory if not existing
if os.path.exists('Fsplit'):
    shutil.rmtree('Fsplit')
os.mkdir('Fsplit')

# Save splits
np.save('./Fsplit/Ftrain.npy', Ftrain)
np.save('./Fsplit/Fval.npy', Fval)
np.save('./Fsplit/Ftest.npy', Ftest)

# Print split counts
print(len(Ftrain), 'folders for training')
print(len(Fval), 'folders for validation')
print(len(Ftest), 'folders for testing')






import numpy as np
import rasterio
import os
import pandas as pd
import shutil

# Set values directly
Fsplit = './Fsplit/'  # Path to Fsplit folder
patch_size = 32  # Patch size
step = 16  # Step for sliding window

Ftrain = np.load(Fsplit + 'Ftrain.npy').tolist()
Fval = np.load(Fsplit + 'Fval.npy').tolist()

def shuffle(vector):
    vector = np.asarray(vector)
    p = np.random.permutation(len(vector))
    vector = vector[p]
    return vector

def sliding_window_train(i_city, labeled_areas, label, window_size, step):
    city = []
    fpatches_labels = []

    x = 0
    while x != label.shape[0]:
        y = 0
        while y != label.shape[1]:

            if not (y + window_size > label.shape[1]) and not (x + window_size > label.shape[0]):
                line = np.array([x, y, labeled_areas.index(i_city)]) 
                city.append(line)

                new_patch_label = label[x:x + window_size, y:y + window_size]
                ff = np.where(new_patch_label == 1)
                if ff[0].shape[0] == 0:
                    stride = window_size
                else:
                    stride = step

            if y + window_size == label.shape[1]:
                break

            if y + window_size > label.shape[1]:
                y = label.shape[1] - window_size
            else:
                y = y + stride

        if x + window_size == label.shape[0]:
            break

        if x + window_size > label.shape[0]:
            x = label.shape[0] - window_size
        else:
            x = x + stride

    return np.asarray(city)

# Clean up the 'xys' directory if it exists
if os.path.exists('xys'):
    shutil.rmtree('xys')
os.mkdir('xys')

# Process the training data
cities = []
for i_city in Ftrain:
    print('train ', i_city)
    path = i_city + '/change/change.tif'

    # Use rasterio to read the image
    with rasterio.open(path) as dataset:
        train_gt = dataset.read(1)  # Read the first band of the TIFF file

    xy_city = sliding_window_train(i_city, Ftrain, train_gt, patch_size, step)
    cities.append(xy_city)

final_cities = np.concatenate(cities, axis=0)
final_cities = shuffle(final_cities)

# Save train xys to CSV
df = pd.DataFrame({'X': list(final_cities[:, 0]),
                   'Y': list(final_cities[:, 1]),
                   'image_ID': list(final_cities[:, 2]),
                   })
df.to_csv('./xys/myxys_train.csv', index=False, columns=["X", "Y", "image_ID"])

# Process the validation data
cities = []
for i_city in Fval:
    print('val ', i_city)
    path = i_city + '/change/change.tif'

    # Use rasterio to read the image
    with rasterio.open(path) as dataset:
        val_gt = dataset.read(1)  # Read the first band of the TIFF file

    xy_city = sliding_window_train(i_city, Fval, val_gt, patch_size, patch_size)
    cities.append(xy_city)

final_cities = np.concatenate(cities, axis=0)
final_cities = shuffle(final_cities)

# Save val xys to CSV
df = pd.DataFrame({'X': list(final_cities[:, 0]),
                   'Y': list(final_cities[:, 1]),
                   'image_ID': list(final_cities[:, 2]),
                   })
df.to_csv('./xys/myxys_val.csv', index=False, columns=["X", "Y", "image_ID"])


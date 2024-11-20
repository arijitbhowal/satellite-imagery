
import glob
import cv2
from skimage import io
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import network
import tools
import shutil

# Define parameters
patch_size = 256
step = 128
saved_model = '/unet_lstm/model_4.pt'

# Function for sliding window
def sliding_window(IMAGE, patch_size, step):
    prediction = np.zeros((IMAGE.shape[3], IMAGE.shape[4], 2))
    count_image = np.zeros((IMAGE.shape[3], IMAGE.shape[4]))
    x = 0
    while x != IMAGE.shape[3]:
        y = 0
        while y != IMAGE.shape[4]:
            if (not y + patch_size > IMAGE.shape[4]) and (not x + patch_size > IMAGE.shape[3]):
                patch = IMAGE[:, :, :, x:x + patch_size, y:y + patch_size] / 255.0
                patch = tools.to_cuda(torch.from_numpy(patch).float())
                output, segm1, segm2 = model(patch)
                output = F.log_softmax(output, dim=1)
                output = output.cpu().data.numpy().squeeze()
                output = np.transpose(output, (1, 2, 0))
                for i in range(patch_size):
                    for j in range(patch_size):
                        prediction[x + i, y + j] += output[i, j, :]
                        count_image[x + i, y + j] += 1

                stride = step

            if y + patch_size == IMAGE.shape[4]:
                break

            if y + patch_size > IMAGE.shape[4]:
                y = IMAGE.shape[4] - patch_size
            else:
                y = y + stride

        if x + patch_size == IMAGE.shape[3]:
            break

        if x + patch_size > IMAGE.shape[3]:
            x = IMAGE.shape[3] - patch_size
        else:
            x = x + stride

    final_pred = np.zeros((IMAGE.shape[3], IMAGE.shape[4]))
    for i in range(final_pred.shape[0]):
        for j in range(final_pred.shape[1]):
            final_pred[i, j] = np.argmax(prediction[i, j] / float(count_image[i, j]))

    return final_pred, prediction

# Load the model
model = tools.to_cuda(network.U_Net(4, 2, 256))
model.load_state_dict(torch.load(saved_model))
model.eval()

# Specify folder for predictions
folder_path = '/river_dataset/L15-0797E-1133N_1133_1139_13/images'  # Replace with your folder path
save_folder = 'PREDICTIONS'

# Ensure save folder exists
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)

# Process all .tif images in the specified folder
all_tifs = glob.glob(os.path.join(folder_path, '*.tif'))
if not all_tifs:
    raise FileNotFoundError(f"No .tif images found in the folder: {folder_path}")

img = []
for tif in tqdm(all_tifs, desc="Loading images"):
    im = io.imread(tif)
    if len(im.shape) == 2:  # Handle grayscale images
        im = np.stack([im] * 4, axis=-1)
    elif im.shape[-1] < 4:  # Handle channels < 4
        im = np.concatenate([im, np.zeros((*im.shape[:-1], 4 - im.shape[-1]))], axis=-1)
    img.append(im)

# Convert to numpy array and ensure dimensions
img = np.asarray(img)
img = np.transpose(img, (0, 3, 1, 2))  # (N, Channels, Height, Width)
imgs = np.expand_dims(img, 1)

# Run sliding window predictions
pred, prob = sliding_window(imgs, patch_size, step)
prob = np.transpose(prob, (2, 0, 1))

# Save predictions
pred_save_path = os.path.join(save_folder, 'Predictions.tif')
prob_save_path = os.path.join(save_folder, 'Probabilities.tif')

io.imsave(pred_save_path, pred.astype(np.uint8))
io.imsave(prob_save_path, prob.astype(np.float32))

print(f"Predictions saved in: {save_folder}")




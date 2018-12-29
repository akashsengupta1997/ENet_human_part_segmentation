import os
import cv2
import pickle
import numpy as np

masks_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks/train"
output_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/1he_labels/train"

NUM_BODYPARTS = 31


def classlab(labels, num_classes):
    x = np.zeros((labels.shape[0], labels.shape[1], num_classes))
    for bodypart in range(num_classes):
        x[labels == bodypart, bodypart] = 1.0
    return x


for mask in sorted(os.listdir(masks_dir)):
    mask_path = os.path.join(masks_dir, mask)
    one_hot_encoded_labels_path = os.path.join(output_dir, os.path.splitext(mask)[0]
                                               + "_OHE.pkl")
    mask = cv2.imread(mask_path, 0)
    one_hot_encoded_labels = classlab(mask, NUM_BODYPARTS)
    with open(one_hot_encoded_labels_path, 'wb') as outfile:
        pickle.dump(one_hot_encoded_labels, outfile)

print("OHE Labels saved to ", output_dir)

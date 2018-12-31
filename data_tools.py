"""
Functions to deal with loading data from different datasets.
Currently only has functions for the Pascal Person Part dataset.
"""

import os
import shutil

from matplotlib import pyplot as plt

IMAGE_PATH_LENGTH = 28


def copy_images_to_PPP(VOC_dir):
    """
    Copy training images specified in pascal_person_part training list
     from VOC folder to pascal_person_part folder.
    :param VOC_dir:
    """

    list_file_path = os.path.join(VOC_dir, 'pascal_person_part/pascal_person_part_trainval_list/trainval.txt')
    print(list_file_path)
    with open(list_file_path, 'r') as list_file:
        for line in list_file:
            relative_image_path = line[1:IMAGE_PATH_LENGTH-1]
            image = os.path.basename(relative_image_path)
            current_image_path = os.path.join(VOC_dir, relative_image_path)
            destination_image_path = os.path.join(VOC_dir,
                                                  'pascal_person_part/train_images/train',
                                                  image)
            shutil.copy(current_image_path, destination_image_path)


def labels_from_seg_image(seg_image):
    """
    PPP seg_images have pixels labelled as follows:
    Lower leg - 89
    Upper leg - 52
    Lower arm - 14
    Upper arm - 113
    Torso - 75
    Head - 38
    Background - 0

    This function changes labels to:
    Lower leg - 1
    Upper leg - 2
    Lower arm - 3
    Upper arm - 4
    Torso - 5
    Head - 6
    Background - 0

    :param seg_image: batch of segmentation masks with Pascal VOC labels
    :return: relabelled batch of segmentation masks
    """
    copy = seg_image.copy()
    copy[seg_image == 89] = 1
    copy[seg_image == 52] = 2
    copy[seg_image == 14] = 3
    copy[seg_image == 113] = 4
    copy[seg_image == 75] = 5
    copy[seg_image == 38] = 6
    return copy


def convert_ppp_labels(source_folder, destination_folder):

    for fname in sorted(os.listdir(source_folder)):
        if fname.endswith(".png"):
            orig_mask = plt.imread(os.path.join(source_folder, fname))
            new_mask = labels_from_seg_image(orig_mask)
            plt.imsave(os.path.join(destination_folder, fname), new_mask)


convert_ppp_labels('/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/'
                   'pascal_person_part/train_masks/train',
                   '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/ppp+up-s31/'
                   'train_masks/train')


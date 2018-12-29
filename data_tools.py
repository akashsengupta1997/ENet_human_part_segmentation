"""
Functions to deal with loading data from different datasets.
Currently only has functions for the Pascal Person Part dataset.
"""

import os
import shutil

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







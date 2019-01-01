import os
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from train_bodyparts import generate_data, classlab


def evaluate_models(model_dir, dataset):
    """
    :param model_dir: directory of model weights
    :param dataset: which dataset to test on
    """
    batch_size = 10

    if dataset == 'ppp' or dataset == 'ppp+up-s31':
        eval_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/' \
                         'pascal_person_part/val_images'
        eval_label_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/' \
                        'pascal_person_part/val_masks'
        num_classes = 7
        num_val_images = 500
    elif dataset == 'up-s31':
        eval_image_dir = None
        eval_mask_dir = None
        num_classes = 32

    eval_image_data_gen_args = dict(
        rescale=(1 / 255.0),
        fill_mode='nearest')

    eval_mask_data_gen_args = dict(
        fill_mode='nearest')

    eval_image_datagen = ImageDataGenerator(**eval_image_data_gen_args)
    eval_mask_datagen = ImageDataGenerator(**eval_mask_data_gen_args)

    for model_name in sorted(os.listdir(model_dir)):
        if model_name.endswith(".hdf5"):
            print(model_name)
            img_wh = 256
            if '256' in model_name:
                img_dec_wh = 256
            elif '64' in model_name:
                img_dec_wh = 64

            model = load_model(os.path.join(model_dir, model_name))

            seed = 1
            eval_image_generator = eval_image_datagen.flow_from_directory(
                eval_image_dir,
                batch_size=batch_size,
                target_size=(img_wh, img_wh),
                class_mode=None,
                seed=seed)

            eval_mask_generator = eval_mask_datagen.flow_from_directory(
                eval_label_dir,
                batch_size=batch_size,
                target_size=(img_dec_wh, img_dec_wh),
                class_mode=None,
                color_mode="grayscale",
                seed=seed)

            def eval_data_gen():
                while True:
                    train_data, train_labels = generate_data(eval_image_generator,
                                                             eval_mask_generator,
                                                             batch_size, num_classes, dataset)
                    reshaped_train_labels = np.reshape(train_labels,
                                                       (batch_size, img_dec_wh * img_dec_wh,
                                                        num_classes))

                    yield (train_data, reshaped_train_labels)

            metrics = model.evaluate_generator(eval_data_gen(),
                                               steps=int(num_val_images/batch_size))

            print(metrics)


evaluate_models('/Users/Akash_Sengupta/Documents/enet/ppp_body_part_models', 'ppp')




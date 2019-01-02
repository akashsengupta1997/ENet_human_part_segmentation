import os
import numpy as np
import cv2

from matplotlib import pyplot as plt

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from train_bodyparts import generate_data, labels_from_seg_image


def evaluate_pixel_accuracy_CC_loss(model_dir, dataset):
    """
    Function to evaluate model pixel accuracy and crossentropy loss on given dataset.
    Model dir is a directory of hdf5 files.
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

            print('model:', model_name, 'loss:', metrics[0], 'pix acc', metrics[1], '\n')


def mean_iou(ground_truth, predict, num_classes):
    """
    Compute IoU averaged over all classes between ground truth label and predicted label
    :param ground_truth:
    :param predict:
    :param num_classes:
    :return:
    """
    class_ious = []
    for class_num in range(1, num_classes):  # not including background class
        ground_truth_binary = np.zeros(ground_truth.shape)
        predict_binary = np.zeros(predict.shape)
        ground_truth_binary[ground_truth == class_num] = 1
        predict_binary[predict == class_num] = 1

        intersection = np.logical_and(ground_truth_binary, predict_binary)
        union = np.logical_or(ground_truth_binary, predict_binary)
        if np.sum(union) != 0:  # Don't include if no occurences of class in image
            iou_score = np.sum(intersection) / np.sum(union)
            class_ious.append(iou_score)

    return np.mean(class_ious)


def evaluate_IoU(model_dir, dataset):

    if dataset == 'ppp' or dataset == 'ppp+up-s31':
        eval_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/' \
                         'pascal_person_part/val_images/val'
        eval_label_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/' \
                         'pascal_person_part/val_masks/val'
        num_classes = 7
        num_val_images = 500
    elif dataset == 'up-s31':
        eval_image_dir = None
        eval_mask_dir = None
        num_classes = 32

    for model_name in sorted(os.listdir(model_dir)):
        image_ious = []
        if model_name.endswith(".hdf5"):
            print(model_name)
            img_wh = 256
            if '256' in model_name:
                img_dec_wh = 256
            elif '64' in model_name:
                img_dec_wh = 64

            model = load_model(os.path.join(model_dir, model_name))

            img_fnames = sorted(os.listdir(eval_image_dir))
            mask_fnames = sorted(os.listdir(eval_label_dir))

            for i in range(len(img_fnames)):
                img_fname = img_fnames[i]
                mask_fname = mask_fnames[i]

                image = cv2.imread(os.path.join(eval_image_dir, img_fname))
                image = cv2.resize(image, (img_wh, img_wh))
                image = image[..., ::-1]
                image = image * (1 / 255.0)

                image_tensor = np.expand_dims(image, 0)

                output = np.reshape(model.predict(image_tensor),
                                     (1, img_dec_wh, img_dec_wh, num_classes))

                labels = output[0, :, :, :]
                predicted_seg = np.argmax(labels, axis=2)
                ground_truth = cv2.imread(os.path.join(eval_label_dir, mask_fname), 0)
                ground_truth = cv2.resize(ground_truth, (img_dec_wh, img_dec_wh),
                                          interpolation=cv2.INTER_NEAREST)
                ground_truth = labels_from_seg_image(ground_truth)
                plt.figure()
                plt.subplot(211)
                plt.imshow(ground_truth, cmap='gray')
                plt.subplot(212)
                plt.imshow(predicted_seg, cmap='gray')
                plt.show()
                iou = mean_iou(ground_truth, predicted_seg, num_classes)
                # print(iou)
                image_ious.append(iou)

            model_iou = np.mean(image_ious)
            print('MODEL IOU', model_name, model_iou)


evaluate_pixel_accuracy_CC_loss('/Users/Akash_Sengupta/Documents/enet/ppp_body_part_models', 'ppp')


# evaluate_IoU('/Users/Akash_Sengupta/Documents/enet/ppp_body_part_models', 'ppp')


import os
from keras.optimizers import SGD
import cv2
import numpy as np
import time


from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.engine.topology import Layer
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, \
    UpSampling2D, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
from matplotlib.pylab import cm
from matplotlib import pyplot as plt
from encoder_enet_simple import build_enet, build_enet_64
from decoder_enet_simple import build_enet_dec, build_enet_dec_64

np.set_printoptions(threshold=np.nan)


def labels_from_seg_image(seg_image_batch):
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

    :param seg_image_batch: batch of segmentation masks with Pascal VOC labels
    :return: relabelled batch of segmentation masks
    """
    copy_batch = seg_image_batch.copy()
    copy_batch[seg_image_batch == 89] = 1
    copy_batch[seg_image_batch == 52] = 2
    copy_batch[seg_image_batch == 14] = 3
    copy_batch[seg_image_batch == 113] = 4
    copy_batch[seg_image_batch == 75] = 5
    copy_batch[seg_image_batch == 38] = 6
    return copy_batch


# TODO: we actually want labels to be H*W x C (this is done later)
# -> why not just do this directly?
def classlab(labels, num_classes):
    """
    Function to convert HxWx1 labels image to HxWxC one hot encoded matrix.
    :param labels: HxWx1 labels image
    :param num_classes: number of segmentation classes
    :return: HxWxC one hot encoded matrix.
    """
    x = np.zeros((labels.shape[0], labels.shape[1], num_classes))
    # print('IN CLASSLAB', labels.shape)
    for pixel_class in range(num_classes):
        indexes = list(zip(*np.where(labels == pixel_class)))
        for index in indexes:
            x[index[0], index[1], pixel_class] = 1.0
    # print("class lab shape", x.shape)
    return x


def generate_data(image_generator, mask_generator, n, num_classes, dataset):
    images = []
    labels = []
    i = 0
    while i < n:
        x = image_generator.next()
        y = mask_generator.next()
        if dataset == 'ppp':  # Need to change labels if using ppp dataset
            y = labels_from_seg_image(y)
        j = 0
        while j < x.shape[0]:
            images.append(x[j, :, :, :])
            labels.append(classlab(y[j, :, :, :].astype(np.uint8), num_classes))
            j = j + 1
            i = i + 1
            if i >= n:
                break

    return np.array(images), np.array(labels)


def build_enet_model(img_wh, img_dec_wh, num_classes):
    inp = Input(shape=(img_wh, img_wh, 3))

    if img_dec_wh == 256:
        enet = build_enet(inp)
        enet = build_enet_dec(enet, nc=num_classes)
    elif img_dec_wh == 64:
        enet = build_enet_64(inp)
        enet = build_enet_dec_64(enet, nc=num_classes)

    enet = Reshape((img_dec_wh * img_dec_wh, num_classes))(enet)
    enet = Activation('softmax')(
        enet)  # softmax is computed for the last dimension - i.e. over classes, as desired
    enet_model = Model(inputs=inp, outputs=enet)

    return enet_model


def segmentation_train(img_wh, img_dec_wh, dataset):
    batch_size = 1  # TODO change back to 10

    if dataset == 'up-s31':
        train_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/images"
        train_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/masks"
        val_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/val_images"
        val_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/val_masks"
        num_classes = 32
        num_train_images = 7664
        num_val_images = 851

    elif dataset == 'ppp':
        train_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/pascal_person_part/train_images'
        train_label_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/pascal_person_part/train_masks'

        val_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/pascal_person_part/val_images'
        val_label_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/pascal_person_part/val_masks'
        num_classes = 7
        num_train_images = 3033
        num_val_images = 500

    elif dataset == 'ppp+up-s31':
        train_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/ppp+up-s31/train_images'
        train_label_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/ppp+up-s31/train_masks'

        val_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/ppp+up-s31/val_images'
        val_label_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/ppp+up-s31/val_masks'
        num_classes = 7
        num_train_images = 7340
        num_val_images = 1000

    assert os.path.isdir(train_image_dir), 'Invalid image directory'
    assert os.path.isdir(train_label_dir), 'Invalid label directory'
    assert os.path.isdir(val_image_dir), 'Invalid validation image directory'
    assert os.path.isdir(val_label_dir), 'Invalid validation label directory'

    train_image_data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1/255.0,
        fill_mode='nearest')

    train_mask_data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_image_data_gen_args = dict(
        rescale=(1/255.0),
        fill_mode='nearest')

    val_mask_data_gen_args = dict(
        fill_mode='nearest')

    train_image_datagen = ImageDataGenerator(**train_image_data_gen_args)
    train_mask_datagen = ImageDataGenerator(**train_mask_data_gen_args)
    val_image_datagen = ImageDataGenerator(**val_image_data_gen_args)
    val_mask_datagen = ImageDataGenerator(**val_mask_data_gen_args)

    # Provide the same seed to flow methods for train generators
    seed = 1
    train_image_generator = train_image_datagen.flow_from_directory(
        train_image_dir,
        batch_size=batch_size,
        target_size=(img_wh, img_wh),
        class_mode=None,
        seed=seed)

    train_mask_generator = train_mask_datagen.flow_from_directory(
        train_label_dir,
        batch_size=batch_size,
        target_size=(img_dec_wh, img_dec_wh),
        class_mode=None,
        color_mode="grayscale",
        seed=seed)

    val_image_generator = val_image_datagen.flow_from_directory(
        val_image_dir,
        batch_size=batch_size,
        target_size=(img_wh, img_wh),
        class_mode=None,
        seed=seed)

    val_mask_generator = val_mask_datagen.flow_from_directory(
        val_label_dir,
        batch_size=batch_size,
        target_size=(img_dec_wh, img_dec_wh),
        class_mode=None,
        color_mode="grayscale",
        seed=seed)

    print('Generators loaded.')

    # For testing data loading
    x = train_image_generator.next()
    y = train_mask_generator.next()
    print('x shape in generate data', x.shape)  # should = (batch_size, img_hw, img_hw, 3)
    print('y shape in generate data', y.shape)  # should = (batch_size, dec_hw, dec_hw, 1)
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(x[0, :, :, :])
    plt.subplot(222)
    plt.imshow(y[0, :, :, 0])
    # y_post = labels_from_seg_image(y)
    # plt.subplot(223)
    # plt.imshow(y_post[0, :, :, 0])
    plt.show()

    enet_model = build_enet_model(img_wh, img_dec_wh, num_classes)
    enet_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print("Model compiled.")

    for trials in range(4000):
        nb_epoch = 1
        print("Fitting", trials)

        def train_data_gen():
            while True:
                train_data, train_labels = generate_data(train_image_generator,
                                                         train_mask_generator,
                                                         batch_size,
                                                         num_classes,
                                                         dataset)
                reshaped_train_labels = np.reshape(train_labels,
                                                   (batch_size, img_dec_wh * img_dec_wh,
                                                    num_classes))

                yield (train_data, reshaped_train_labels)

        def val_data_gen():
            while True:
                val_data, val_labels = generate_data(val_image_generator,
                                                     val_mask_generator,
                                                     batch_size,
                                                     num_classes)
                reshaped_val_labels = np.reshape(val_labels,
                                                   (batch_size, img_dec_wh * img_dec_wh,
                                                    num_classes))
                yield (val_data, reshaped_val_labels)

        history = enet_model.fit_generator(train_data_gen(),
                                            steps_per_epoch=int(num_train_images/batch_size),
                                            nb_epoch=nb_epoch,
                                            verbose=1,
                                            validation_data=val_data_gen(),
                                            validation_steps=int(num_val_images/batch_size))

        print("After fitting")
        if trials % 200 == 0:

            # Monitor training
            img_list = []
            fnames = []
            monitor_images_dir = "./monitor_train/monitor_train_images"

            for fname in sorted(os.listdir(monitor_images_dir)):
                if fname.endswith(".png"):
                    image = cv2.imread(os.path.join(monitor_images_dir, fname))
                    image = cv2.resize(image, (img_wh, img_wh))
                    image = image[..., ::-1]
                    img_list.append(image / 255.0)
                    fnames.append(os.path.splitext(fname)[0])

            img_tensor = np.array(img_list)
            output = np.reshape(enet_model.predict(img_tensor),
                                (-1, img_dec_wh, img_dec_wh,
                                 num_classes))

            for img_num in range(len(img_list)):
                seg_labels = output[img_num, :, :, :]
                seg_img = np.argmax(seg_labels, axis=-1)
                plt.figure(1)
                plt.clf()
                plt.imshow(seg_img)
                plt.savefig("./monitor_train/seg_" + str(trials) + "_" + fnames[img_num] +
                            "_seg_image.png")

            # Save model
            enet_model.save('up-s31_body_part_models/enet256_' + str(trials + 1).zfill(4)
                            + '.hdf5')

    print("Finished")


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
segmentation_train(256, 256, 'up-s31')

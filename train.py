import os
from keras.optimizers import SGD
import cv2
import numpy as np
import time


np.random.seed(123)  # for reproducibility
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
import itertools
from keras.preprocessing.image import Iterator
from encoder_enet_simple import build_enet, build_enet_64
from decoder_enet_simple import build_enet_dec, build_enet_dec_64

DETECT_TIMES = []


def normalized(rgb):
    # return rgb
    norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)
    b = rgb[:, :, 0]
    g = rgb[:, :, 1]
    r = rgb[:, :, 2]

    norm[:, :, 0] = cv2.equalizeHist(b)
    norm[:, :, 1] = cv2.equalizeHist(g)
    norm[:, :, 2] = cv2.equalizeHist(r)
    return norm


def binarylab(labels):
    # print(str(labels))
    x = np.zeros((labels.shape[0], labels.shape[1], 2))
    x[labels == 0, 0] = 1.0
    x[labels > 0, 1] = 1.0
    return x


def prep_data(images_dir, train_file_name):
    train_data = []
    train_label = []
    with open(train_file_name) as f:
        txt = f.readlines()
    txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        image_file_name = images_dir + txt[i][0]
        label_file_name = images_dir + txt[i][1].split('\n')[0]
        # print("IM = "+image_file_name+"|")
        assert (os.path.isfile(image_file_name)), "Image file does not exist"

        # print("LA = "+label_file_name+"|")
        assert (os.path.isfile(label_file_name)), "Label file does not exist"

        # train_data.append(np.rollaxis(normalized(cv2.imread(image_file_name)),2))
        train_data.append(normalized(cv2.imread(image_file_name)))
        train_label.append(binarylab(cv2.imread(label_file_name, 0)))
    return np.array(train_data), np.array(train_label)


def create_encoding_layers(img_wh):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    kernel_initializer = 'glorot_normal'
    bias_initializer = 'random_uniform'
    return [
        ZeroPadding2D(padding=(pad, pad), input_shape=(img_wh, img_wh, 3)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid',
                      kernel_initializer=kernel_initializer),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid',
                      kernel_initializer=kernel_initializer),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid',
                      kernel_initializer=kernel_initializer),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid',
                      kernel_initializer=kernel_initializer),
        BatchNormalization(),
        Activation('relu'),
        # MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]


def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    kernel_initializer = 'glorot_normal'
    bias_initializer = 'random_uniform'
    return [
        # UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid',
                      kernel_initializer=kernel_initializer),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid',
                      kernel_initializer=kernel_initializer),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid',
                      kernel_initializer=kernel_initializer),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid',
                      kernel_initializer=kernel_initializer),
        BatchNormalization(),
    ]


def test(train_data, model, ep, img_wh, img_dec_wh, image_dir_list):
    img_list = []
    if not (train_data is None):
        for id in range(0, 10):
            # plt.imshow((train_data[0][id,:,:,:]).astype(np.uint8))
            # plt.show()
            img_list.append(train_data[0][id, :, :, :])
    # plt.imshow((img_list[id]).astype(np.uint8))
    # plt.show()

    for image_dir in image_dir_list:
        for fname in sorted(os.listdir(image_dir)):
            image = cv2.imread(image_dir + fname)
            # plt.imshow(image)
            # plt.show()

            # # This code is for timing predicts for single images-comment out in actual tests
            # time_test_image_tensor = np.array(image)
            # time_test_image_tensor = np.expand_dims(time_test_image_tensor, 0)
            # start = time.time()
            # time_test_output = model.predict(time_test_image_tensor)
            # end = time.time()
            # DETECT_TIMES.append(end - start)
            # print('detect time:', end-start)

            img_list.append(image / 255.0)

    # print("Number of test images", len(DETECT_TIMES) - 1, "Average detect time:",
    #       np.mean(DETECT_TIMES[1:]))

    img_tensor = np.array(img_list)
    output = np.reshape(model.predict(img_tensor), (len(img_list), img_dec_wh, img_dec_wh, 2))
    # print('s = '+str(output.shape))
    results_dir = 'images-ex04/results64/' + 'ep_' + str(ep).zfill(4) + '/'
    os.system('mkdir ' + results_dir)

    for id in range(0, len(img_list)):
        # print(str(output[id,:,:,0]))
        cv2.imwrite(
            results_dir + 'img_ep' + str(ep).zfill(4) + '_' + str(id).zfill(5) + '_lab.png',
            (255 * output[id, :, :, 1]))
        cv2.imwrite(
            results_dir + 'img_ep' + str(ep).zfill(4) + '_' + str(id).zfill(5) + '.png',
            (255 * img_list[id]).astype(np.uint8))


# plt.imshow((255*img_list[id]).astype(np.uint8))
# plt.draw()
# plt.show(block=False)


def generate_data(generator, n, batch_size):
    images = []
    labels = []
    i = 0
    while i < n:
        x, y = generator.next()
        # plt.imshow(y[0, :, :, 0] * 8)
        # plt.show()
        # print(str(len(images))+ " " +str(x.shape))
        j = 0

        while j < x.shape[0]:
            images.append(x[j, :, :, :])
            labels.append(binarylab(y[j, :, :, 0].astype(np.uint8)))
            j = j + 1
            i = i + 1
            if (i >= n):
                break
    return np.array(images), np.array(labels)


def random_noise(data, sync_seed=None, **kwargs):
    print(str(data.shape))
    return data


def segmentation_train():
    img_wh = 256
    img_dec_wh = 64
    batch_size = 10
    project_name = 'mpii-' + str(img_wh)
    test_image_dir_list = []
    test_image_dir_list.append('images-ex04/ben-vid690-' + str(img_wh) + '/')
    test_image_dir_list.append('images-ex04/fotis-vid489-' + str(img_wh) + '/')

    # train_image_dir = 'images-ex04/' + project_name + '/images'
    # train_label_dir = 'images-ex04/' + project_name + '/labels'
    train_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31/images"
    train_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31/masks"
    assert os.path.isdir(train_image_dir), 'Invalid image directory'
    assert os.path.isdir(train_label_dir), 'Invalid label directory'
    print('images = ' + str(len(os.listdir(train_image_dir))))
    # train_data, train_label_images = prep_data('images-ex04/'+project_name+'/','images-ex04/'+project_name+'/train.txt')
    # train_label = np.reshape(train_label_images,(13030,img_wh*img_wh,2))

    data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1 / 255.0,
        fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    #	image_datagen.fit(image_datagen, augment=True, seed=seed)
    #	mask_datagen.fit(mask_datagen, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        train_image_dir,
        batch_size=batch_size,
        target_size=(img_wh, img_wh),
        class_mode=None,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_label_dir,
        batch_size=batch_size,
        # target_size=(img_wh, img_wh),
        target_size=(img_dec_wh, img_dec_wh),
        # train_image_dir,
        class_mode=None,
        seed=seed)

    # combine generators into one which yields image and masks
    print('before gen')
    train_generator = zip(image_generator, mask_generator)
    # train_generator = itertools.izip(image_generator, mask_generator)
    print('got generator')
    # x, y = train_generator.next()
    x = image_generator.next()
    y = mask_generator.next()
    plt.imshow(y[0, :, :, 0] * 8)
    plt.show()
    print('x shape ' + str(x.shape))
    print('y shape ' + str(y.ndim))
    print('got data')
    for i in range(0, 5):
        comb = np.concatenate(
            (cv2.resize(255 * x[i, :, :, :], (img_dec_wh, img_dec_wh)), 255 * y[i, :, :, :]),
            axis=1)
        # comb=x[i,:,:,:]
        # plt.imshow(comb.astype(np.uint8))
        cv2.imwrite('a_' + str(i).zfill(4) + '.png', comb.astype(np.uint8))
    # plt.figure()
    # plt.subplot(211)
    # plt.imshow(x[i,:,:,:].squeeze().astype(np.uint8))
    # plt.subplot(221)
    # plt.imshow(y[i,:,:,:])
    # plt.draw()
    # plt.show(block=False)

    class_weighting = [0.02, 0.98]

    # autoencoder = Sequential()

    # autoencoder.encoding_layers = create_encoding_layers(img_wh)
    # autoencoder.decoding_layers = create_decoding_layers()
    # for l in autoencoder.encoding_layers:
    #   autoencoder.add(l)
    # for l in autoencoder.decoding_layers:
    #   autoencoder.add(l)

    # autoencoder.add(Convolution2D(2, 1, 1, border_mode='valid'))

    nc = 2  # Number of classes

    inp = Input(shape=(img_wh, img_wh, 3))
    # enet = build_enet(inp)
    # enet = build_enet_dec(enet, nc=nc)
    enet = build_enet_64(inp)
    enet = build_enet_dec_64(enet, nc=nc)
    name = 'enet_unpooling'

    enet = Reshape((img_dec_wh * img_dec_wh, nc))(
        enet)  # TODO: need to remove data_shape for multi-scale training
    enet = Activation('softmax')(enet)
    autoencoder = Model(inputs=inp, outputs=enet)

    #	autoencoder.add(Reshape((img_dec_wh*img_dec_wh,2), input_shape=(img_dec_wh,img_dec_wh,2)))
    #	autoencoder.add(Activation('softmax'))

    ##from keras.optimizers import SGD

    optimizer = SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=False)

    # if(os.path.isfile('model_weight_ep_0010.hdf5')):
    #	autoencoder=load_model('model_weight_ep_0010.hdf5')
    #		print('MOdel loaded from file! ')

    print("Before compiling")
    # autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=['accuracy'])
    autoencoder.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    print("After compiling")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "autoencoder.png")
    # plot(model_path, to_file=model_path, show_shapes=True)

    for trials in range(11, 4000):
        print('Before generating augmented data')
        # train_data, train_label_images = generate_data(train_generator,13030,320)
        # lab=(255*np.asarray(np.dstack((train_label_images[0][:,:,1], train_label_images[0][:,:,1], train_label_images[0][:,:,1])))).astype(np.uint8)
        # print('lab = '+str(lab.shape)+' t = '+str(train_data[0].shape))
        # comb=np.concatenate((train_data[0],lab), axis=1)
        # plt.imshow(comb.astype(np.uint8))
        # plt.show()

        # train_label = np.reshape(train_label_images,(13030,img_wh*img_wh,2))
        print('After generating augmented data')
        nb_epoch = 1

        print("Fitting")

        # history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch,  verbose=1, class_weight=class_weighting )#, validation_data=(X_test, X_test))
        def data_gen():
            while True:
                train_data, train_label_images = generate_data(train_generator, batch_size,
                                                               batch_size)
                train_label = np.reshape(train_label_images,
                                         (batch_size, img_dec_wh * img_dec_wh, 2))
                # plt.imshow((train_data[0,:,:,:]).astype(np.uint8))
                # plt.show()
                yield (train_data, train_label)

        history = autoencoder.fit_generator(data_gen(), samples_per_epoch=1000,
                                            nb_epoch=nb_epoch, verbose=1,
                                            class_weight=class_weighting)
        print("After fitting")
        autoencoder.save(
            'e64_model_weight_ep_' + str(nb_epoch * (trials + 1)).zfill(4) + '.hdf5')
        # test([train_data,train_label_images],autoencoder,nb_epoch*(trials+1),img_wh,test_image_dir_list)

        train_data_batch, train_label_images_batch = generate_data(train_generator, batch_size,
                                                                   batch_size)
        train_label_batch = np.reshape(train_label_images_batch,
                                       (batch_size, img_dec_wh * img_dec_wh, 2))
        test([train_data_batch, train_label_batch], autoencoder, nb_epoch * (trials + 1),
             img_wh, img_dec_wh, test_image_dir_list)

    print("Finished")


def segmentation_test(img_wh, img_dec_wh, ep_list):
    # test_image_dir_list = ['images-ex04/fotis-vid489-' + str(img_wh) + '/',
    #                        'images-ex04/ben-vid690-' + str(img_wh) + '/']
    # test_image_dir_list = ['images-ex04/fotis-vid489-' + str(img_wh) + '/']
    test_image_dir_list = ['images-ex04/temp/']
    print('Preloaded model')
    for ep in ep_list:
        autoencoder = load_model('learned_models/model_weight_ep_' + str(ep).zfill(4) + '.hdf5')
        test(None, autoencoder, ep, img_wh, img_dec_wh, test_image_dir_list)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# segmentation_train() #model_weight_ep_0206
# segmentation_test(256, 256,
#                   [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
#                    1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
#                    2700, 2800, 2900, 3000])
# segmentation_test(256, 256,
#                   [3500])

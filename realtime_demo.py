import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from keras.models import load_model


def realtime_demo(img_wh, img_dec_wh, num_classes):
    autoencoder = load_model('/Users/Akash_Sengupta/Documents/enet/ppp_body_part_models/'
                             'enet64_weight0401.hdf5')
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, orig_img = cap.read()
        img = cv2.resize(orig_img, (img_wh, img_wh))
        img = img[..., ::-1]
        img = img * (1/255.0)

        # Add batch dimension: 1 x D x D x 3
        img_tensor = np.expand_dims(img, 0)

        output = np.reshape(autoencoder.predict(img_tensor),
                            (1, img_dec_wh, img_dec_wh, num_classes))

        # Get mask
        seg_labels = output[0, :, :, :]
        seg_img = np.argmax(seg_labels, axis=2)

        # Display
        display_img = cv2.resize(img, (512, 512))
        plt.figure(1)
        plt.imshow(seg_img)
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()
        cv2.imshow('img', display_img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

realtime_demo(256, 64, 7)
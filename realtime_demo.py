import cv2
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from keras.models import load_model


def realtime_demo(img_wh):
    model_path = '/Users/Akash_Sengupta/Documents/enet/ppp+up-s31_body_part_models/enet64_weight1301.hdf5'
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)

    if "64" in model_path:
        img_dec_wh = 64
    elif "256" in model_path:
        img_dec_wh = 256

    if "ppp" in model_path:
        num_classes = 7

    predict_times = []

    while True:
        # Capture frame-by-frame
        ret, orig_img = cap.read()
        bgr_img = cv2.resize(orig_img, (img_wh, img_wh))
        img = bgr_img[..., ::-1]
        img = img * (1/255.0)

        # Add batch dimension: 1 x D x D x 3
        img_tensor = np.expand_dims(img, 0)

        start = time.time()
        raw_output = model.predict(img_tensor)
        predict_times.append(time.time() - start)
        print('Average prediction time:', np.mean(predict_times))

        output = np.reshape(raw_output,(1, img_dec_wh, img_dec_wh, num_classes))

        # Get mask
        seg_labels = output[0, :, :, :]
        seg_img = np.argmax(seg_labels, axis=2)

        # Display
        display_img = cv2.flip(cv2.resize(bgr_img, (512, 512)), 1)
        plt.figure(1)
        plt.imshow(seg_img)
        plt.show(block=False)
        plt.pause(0.05)
        plt.clf()
        cv2.imshow('img', display_img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

realtime_demo(256)
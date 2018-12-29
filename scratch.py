import cv2
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)
image = cv2.imread("/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images/train/00000_image.png")
mask = cv2.imread("/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks/train/00000_ann.png",0)
flattened_mask = np.reshape(mask, (mask.shape[0]*mask.shape[1],))
print('num labels', (np.unique(flattened_mask)).shape)

# image = cv2.resize(image, (256, 256))
# mask = cv2.resize(mask, (64, 64))

image = cv2.resize(image, (256, 256))
mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST)

width, height = mask.shape[0], mask.shape[1]

# There are 32 classes per pixel - 0 is background, 1-31 is bodyparts
num_classes = 32
labels = np.zeros((width, height, num_classes))
# print('IN CLASSLAB', labels.shape)
for pixel_class in range(num_classes):
    indexes = list(zip(*np.where(mask == pixel_class)))
    for index in indexes:
        labels[index[0], index[1], pixel_class] = 1.0

print("labels shape", labels.shape)

plt.figure(1)
plt.clf()
plt.imshow(image)
plt.figure(2)
plt.clf()
plt.subplot(331)
plt.imshow((labels[:, :, 0] * 255), cmap="gray")
plt.subplot(332)
plt.imshow((labels[:, :, 1] * 255), cmap="gray")
plt.subplot(333)
plt.imshow((labels[:, :, 2] * 255), cmap="gray")
plt.subplot(334)
plt.imshow((labels[:, :, 3] * 255), cmap="gray")
plt.subplot(335)
plt.imshow((labels[:, :, 4] * 255), cmap="gray")
plt.subplot(336)
plt.imshow((labels[:, :, 20] * 255), cmap="gray")
plt.subplot(337)
plt.imshow((labels[:, :, 21] * 255), cmap="gray")
plt.subplot(338)
plt.imshow((labels[:, :, 22] * 255), cmap="gray")
plt.subplot(339)
plt.imshow((labels[:, :, 23] * 255), cmap="gray")
plt.show()

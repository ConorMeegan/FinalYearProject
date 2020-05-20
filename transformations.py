import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from scipy import ndimage


def data_load(path):
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    cloud_images = []

    for file in file_list:
        image = mpimg.imread(path + file)
        cloud_images.append(image)

    return cloud_images


def rotation(input_images):
    rotated_90 = []
    rotated_180 = []
    rotated_270 = []
    overall = []

    for image in input_images:
        img_90 = ndimage.rotate(image, angle=90)
        img_180 = ndimage.rotate(image, angle=180)
        img_270 = ndimage.rotate(image, angle=270)

        rotated_90.append(img_90)
        rotated_180.append(img_180)
        rotated_270.append(img_270)

        overall.append(image)
        overall.append(img_90)
        overall.append(img_180)
        overall.append(img_270)

    return rotated_90, rotated_180, rotated_270, overall


def reflection(input_images):
    overall = []

    for image in input_images:
        vertically_reflected_img = np.flip(image, 0)
        horizontally_reflected_img = np.flip(image, 1)
        overall.append(image)
        overall.append(vertically_reflected_img)
        overall.append(horizontally_reflected_img)

    return overall


def save_images(input_images):
    for image in range(np.array(input_images).shape[0]):
        plt.imsave("saved_images/%d.jpg" % (image + 1), input_images[image])

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from scipy import ndimage
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import clustering


class GAN():
    def __init__(self):
        self.img_rows = 500
        self.img_columns = 500
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_columns, self.channels)
        self.latent_dim = 250

        optimizer = Adam(0.00025, 0.5)  # learning rate of 0.00025

        # create the Generator and Discriminator
        self.generator = self.define_generator()
        noise = Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        # loss function to optimise G is min(log(1-D)), in practise we use max(log(D))
        # need to implement this loss
        self.discriminator = self.define_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.discriminator.trainable = False  # only training generator

        valid_input = self.discriminator(img)  # determine whether image is real or fake

        # overall GAN with generator and discriminator combined
        self.combined = Model(noise, valid_input)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def define_generator(self):
        model = Sequential()

        model.add(Dense(32, input_shape=(self.latent_dim, )))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.9))

        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.9))
        #
        # model.add(Dense(128))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.9))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim, ))
        generated_image = model(noise)

        return Model(noise, generated_image)

    def define_discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))  # Flattens input without affecting batch size

        # model.add(Dense(128))
        # model.add(LeakyReLU(alpha=0.2))
        #
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        discriminated_image = Input(shape=self.img_shape)
        valid = model(discriminated_image)

        return Model(discriminated_image, valid)

    def train(self, cloud_images, epochs, batch_size, interval):
        # normalisation of images between [-1, 1]
        # using formula: 2 * (x - x.min()) / (x.max() - x.min()) - 1
        # x.min() being 0, and x.max() being 255 for RGB images
        cloud_images = np.array(cloud_images)
        cloud_images = cloud_images.astype('float32')
        cloud_images = cloud_images/127.5 - 1

        num_train = cloud_images.shape[0]

        real = np.ones((batch_size, 1))  # ground truth array of 1's for real, 0's for fake
        fake = np.zeros((batch_size, 1))
        total_batches = 0

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))
            num_batches = int(np.ceil(num_train/batch_size))
            progress_bar = Progbar(target=num_batches)

            for i in range(num_batches):
                real_image_indices = np.random.randint(0, cloud_images.shape[0], batch_size)
                real_image_batch = cloud_images[real_image_indices]

                # generate images to try and fool the discriminator with
                disc_training_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                disc_generated_images = self.generator.predict(disc_training_noise)

                disc_loss_real = self.discriminator.train_on_batch(real_image_batch, real)
                disc_loss_fake = self.discriminator.train_on_batch(disc_generated_images, fake)

                total_discriminator_loss = np.add(disc_loss_real, disc_loss_fake) / 2

                gen_training_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_loss = self.combined.train_on_batch(gen_training_noise, real)

                print("\nEpoch:", (epoch+1), "\nDiscriminator Loss:", total_discriminator_loss[0],
                      "\nAccuracy:", total_discriminator_loss[1] * 100, "\nGenerator Loss:", gen_loss)

                if total_batches % interval == 0:
                    self.generate_and_save_images(total_batches)

                total_batches = total_batches + 1
                progress_bar.update(i+1)

    def generate_and_save_images(self, epoch):
        row_images, column_images = 1, 1  # 1, 1
        # number of images we generate is equal to --> row_images * column_images = 16
        gen_noise = np.random.normal(0, 1, (row_images*column_images, self.latent_dim))
        gen_images = self.generator.predict(gen_noise)

        # rescaling images
        gen_images = gen_images/2 + 0.5
        figure = plt.figure(figsize=(row_images, column_images))

        for i in range(gen_images.shape[0]):
            plt.subplot(row_images, column_images, i+1)
            # gen_images[i, :, :, 0] * 127.5 + 127.5 normalising in loop
            plt.imshow(gen_images[i, :, :, 0] * 127.5 + 127.5, cmap='Greys')
            plt.axis('off')

        figure.savefig("generated_images/%d.png" % epoch)
        plt.close()


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


if __name__ == '__main__':
    # Used to perform the various affine transformations on the image dataset
    cloud_images = data_load('images/')
    clouds_90, clouds_180, clouds_270, clouds_rotated_overall = rotation(cloud_images)
    overall_cloud_images = reflection(clouds_rotated_overall)
    save_images(overall_cloud_images)

    transformed_cloud_images = data_load('saved_images/')
    gan = GAN()
    gan.train(transformed_cloud_images, epochs=1000, batch_size=32, interval=10)

    # used to generate the binary maps for a sample of images
    image_loc = './generated_single_images_and_binary_maps/43900.png'
    clustering.run_kmeans(image_loc, 43900)

    for i in range(0, 44000, 1000):
        image_location = "./generated_single_images_and_binary_maps/" + str(i) + ".png"
        clustering.run_kmeans(image_location, i)

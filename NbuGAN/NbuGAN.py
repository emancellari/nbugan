from __future__ import print_function

import time
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, LeakyReLU, Conv2D, Conv2DTranspose, \
    BatchNormalization
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications.resnet import decode_predictions

import os, cv2
from PIL import Image
import numpy as np
from numpy.random import seed

class NbuGAN():

    def __init__(self):
        inputs = Input(shape=(224, 224, 3))

        optimizer_g = Adam(0.0002)
        optimizer_d = SGD(0.01)

        generator = self.build_generator(inputs)
        self.G = Model(inputs, generator)
        self.G._name = 'Generator'
        self.G.summary()

        # Build discriminator and train it
        discriminator = self.build_discriminator2(self.G(inputs))
        self.D = Model(inputs, discriminator)
        self.D.compile(loss=tensorflow.keras.losses.binary_crossentropy, optimizer=optimizer_d,
                       metrics=[self.custom_acc])
        self.D._name = 'Discriminator'
        self.D.summary()

        self.target = ResNet50(weights='imagenet')
        self.target.trainable = False

        # Build GAN: stack generator, discriminator and target
        img = (self.G(inputs) / 2 + 0.5) * 255

        # Image is now preprocessed before being fed to VGG16
        self.stacked = Model(inputs=inputs, outputs=[self.G(inputs),
                                                     self.D(inputs), self.target(preprocess_input(img))])

        self.stacked.compile(loss=[self.generator_loss, tensorflow.keras.losses.binary_crossentropy,
                                   tensorflow.keras.losses.categorical_crossentropy], optimizer=optimizer_g)
        self.stacked.summary()

    def generator_loss(self, y_true, y_pred):
        return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - 0.3, 0), axis=-1)  # Hinge loss

    def custom_acc(self, y_true, y_pred):
        return binary_accuracy(K.round(y_pred), K.round(y_true))

    # Basic classification model
    def build_discriminator(self, inputs):
        D = Conv2D(32, 4, strides=(2, 2))(inputs)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)

        D = Conv2D(64, 4, strides=(2, 2))(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Flatten()(D)

        D = Dense(64)(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dense(1, activation='sigmoid')(D)
        return D

    def build_discriminator2(self, inputs):
        D = Conv2D(64, 3, strides=(2, 2), padding='same')(inputs)
        D = LeakyReLU()(D)

        D = Conv2D(128, 3, strides=(2, 2), padding='same')(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)  # Activation function

        D = Conv2D(256, 3, strides=(2, 2), padding='same')(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)

        D = Conv2D(512, 3, strides=(2, 2), padding='same')(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)

        D = Conv2D(512, 3, strides=(2, 2), padding='same')(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)

        D = Flatten()(D)
        D = Dense(512)(D)
        D = LeakyReLU()(D)
        D = Dense(1, activation='sigmoid')(D)

        return D

    def build_generator(self, generator_inputs):

        print("Generator input shape is: ", generator_inputs.shape)

        G = Conv2D(64, 3, padding='same')(generator_inputs)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        G = Conv2D(128, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        G = Conv2D(256, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)
        residual = G

        for _ in range(4):
            G = Conv2D(256, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = Activation('relu')(G)
            G = Conv2D(256, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = layers.add([G, residual])
            residual = G

        G = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        G = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)

        G = Conv2D(3, 3, padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('tanh')(G)

        G = layers.add([G * epsilon / 255, generator_inputs])

        return G

    def train_discriminator(self, x_batch, Gx_batch):
        self.D.trainable = True
        d_loss_real = self.D.train_on_batch(x_batch, np.random.uniform(0.9, 1.0, size=(
            len(x_batch), 1)))
        d_loss_fake = self.D.train_on_batch(Gx_batch, np.zeros((len(Gx_batch), 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        return d_loss

    def train_generator(self, x_batch):
        arr = np.zeros(1000)
        arr[c_t_inx] = 1
        full_target = np.tile(arr, (len(x_batch), 1))
        self.D.trainable = False
        self.target.trainable = False
        stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), full_target])
        return stacked_loss

    def train_GAN(self):
        b0 = time.time()
        img = cv2.imread(clean_image)
        height, width = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_train = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        x_train = np.expand_dims(x_train, axis=0)
        x_train = np.array(x_train, dtype=np.float32)
        x_train = (x_train * 2. / 255 - 1).reshape(len(x_train), 224, 224, 3)

        epochs = EPOCH
        f_90 = open(f"Report.txt", "a")
        for epoch in range(epochs):
            print("===========================================")
            print("EPOCH: ", epoch)
            Gx = self.G.predict(x_train)
            Gx = np.clip(Gx, -1, 1)

            self.train_discriminator(x_train, Gx)
            self.train_generator(x_train)
            np.save('tempAdv/advers_tentative.npy', Gx)

            img_normalized = np.load("tempAdv/advers_tentative.npy").copy()
            img = (img_normalized / 2.0 + 0.5) * 255
            image = img.reshape((1, 224, 224, 3))
            Image.fromarray((image[0]).astype(np.uint8)).save("tempAdv/adv_tentative.png", 'png')

            og = cv2.imread(clean_image)
            og = cv2.cvtColor(og, cv2.COLOR_BGR2RGB)
            og_224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)

            # Extract noise: final image - original image
            noise = img[0] - og_224
            noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LANCZOS4)

            # Create final image: original image + scaled noise and save it in png and npy formats
            final = og + noise
            final = np.clip(final, 0, 255)

            # Re-open HR adversarial image, scale it down to 224x224 for CNN and save it
            final_R = cv2.resize(final, (224, 224), interpolation=cv2.INTER_LANCZOS4)

            image = final_R.reshape((1, 224, 224, 3))
            yhat = self.target.predict(preprocess_input(image))
            pred_labels = decode_predictions(yhat)

            # Report max probability class and target class probability
            pred_max = pred_labels[0][0]
            dom_cat_HR = pred_max[1]
            dom_cat_prop_HR = pred_max[2]
            for label in pred_labels[0]:
                if label[1] == c_t:
                    print(label[1], label[2])
            print(pred_max[1], pred_max[2])

            # Save image if successful and exit; terminate if max epochs reached
            if (np.argmax(yhat, axis=1) == c_t_inx and pred_max[2] >= 0.90) or epoch == epochs - 1:
                e3 = time.time()
                hr_90 = "{:.1f}".format(e3 - b0)

                dom_cat_prop_HR_90 = "{:.4f}".format(dom_cat_prop_HR)
                filename1 = f"advers_{dom_cat_HR}_{dom_cat_prop_HR_90}_{epoch}.npy"
                np.save(filename1, final)
                filename3 = f"advers_{dom_cat_HR}_{dom_cat_prop_HR_90}_{epoch}.png"
                img = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
                cv2.imwrite(filename3, img)
                f_90.write(f"{clean_image} \t{dom_cat_HR} \t\t{dom_cat_prop_HR_90} \t\t{epoch} \t\t {hr_90}\n")
                print("Adversarial image is generated! After %d epochs." % epoch)
                print(f"It took {hr_90} seconds.")
                break

# NbuGAN settings
EPOCH = 1000
epsilon = 4  # maximum magnitude of pixel change [-4:4]
clean_image = '11.JPEG'   # path of the clean image
c_t = 'rhinoceros_beetle'  # target label
T_ct = 0.90    # target label value
c_t_inx = 306  # target category index number

if __name__ == '__main__':
    seed(5)
    tensorflow.random.set_seed(1)
    attack = NbuGAN()
    attack.train_GAN()



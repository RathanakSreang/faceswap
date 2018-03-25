from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import Activation, Dense, Input
from keras.models import Model
from keras import backend as K
from libs.read_images_from import read_images_from
from keras.optimizers import Adam
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="",
    help="model name", required=True)
args = vars(ap.parse_args())

image_size = 64
input_shape = (image_size, image_size, 3)
layer_filters = [32, 64]
kernel_size = 5
latent_dim = 16
batch_size = 128

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs

for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

shape = K.int_shape(x)

x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Build the Decoder Model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)
    x = UpSampling2D((2, 2))(x)


x = Conv2DTranspose(filters=3,
                    kernel_size=kernel_size,
                    padding='same')(x)
outputs = Activation('sigmoid', name='decoder_output')(x)

# Instantiate Decoder Model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()
optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
autoencoder.compile(loss='mean_absolute_error', optimizer=optimizer)

wraped_face, a_faces = read_images_from("images/the_rock")

a_faces = a_faces.astype('float32') / 255.
wraped_face = wraped_face.astype('float32') / 255.

# print(a_faces[0].shape)
# print(wraped_face[0].shape)
# cv2.imshow("wrap image", wraped_face[0])
# cv2.imshow("face image", a_faces[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# b_faces = read_images_from("images/rathanak")
# b_faces = b_faces.astype('float32') / 255.

from keras.callbacks import TensorBoard
autoencoder.fit(wraped_face,
                a_faces,
                epochs=1000,
                batch_size=batch_size,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

autoencoder.save("model/{0}_model.h5".format(args["model"]))
autoencoder.save_weights("model/{0}e_weight.h5".format(args["model"]))

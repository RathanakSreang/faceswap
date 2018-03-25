from keras.models import load_model
from libs.read_images_from import read_images_from
import numpy as np
import cv2
import argparse

image_size = 64
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="",
    help="model name", required=True)
args = vars(ap.parse_args())

autoencoder = load_model("model/{0}_model.h5".format(args["model"]))
autoencoder.load_weights("model/{0}_weight.h5".format(args["model"]))

wrap ,a_faces = read_images_from("images/tests")

# show original image
for (index, img) in enumerate(a_faces):
  cv2.imshow("original_image_" + str(index), img)

a_faces = a_faces.astype('float32') / 255.
wrap = wrap.astype('float32') / 255.

decoded_imgs = autoencoder.predict(a_faces)
decoded_imgs = (decoded_imgs * 255).astype(np.uint8)
for (index, img) in enumerate(decoded_imgs):
  cv2.imshow("swap_image_" + str(index), img)

cv2.waitKey(0)
cv2.destroyAllWindows()

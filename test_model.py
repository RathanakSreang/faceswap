from keras.models import load_model
from libs.read_images_from import read_images_from
import numpy as np
import cv2
image_size = 64

autoencoder = load_model('model/a_face_model.h5')
autoencoder.load_weights('model/a_face_weight.h5')

wrap ,a_faces = read_images_from("images/rathanak")
a_faces = a_faces.astype('float32') / 255.
a_faces = np.reshape(a_faces, (len(a_faces), image_size, image_size, 3))

decoded_imgs = autoencoder.predict(a_faces)

print(decoded_imgs[0])
decoded_imgs = (decoded_imgs * 255).astype(np.uint8)
print(decoded_imgs[0])
for (index, img) in enumerate(decoded_imgs):
  # img = img.reshape(image_size,image_size,3)
  # img = img * 255.
  cv2.imshow("image_name" + str(index), img)

cv2.waitKey(0)
cv2.destroyAllWindows()
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)

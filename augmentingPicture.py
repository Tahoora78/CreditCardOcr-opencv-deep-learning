import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

#this code produces 50 augmented picture and then save it.
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.15, zoom_range=0.1,
    channel_shift_range=10., horizontal_flip=True)
image_path = "path of image which you want to make the augmented image from"
image = np.expand_dims(plt.imread(image_path),0)
plt.imshow(image[0])
aug_iter = gen.flow(image, seed = 0)
aug_images1 = [next(aug_iter)[0].astype(np.uint8) for i in range(50)]
aug_images1 = np.array(aug_images1)

i=0
for im in (aug_images1):
    im = Image.fromarray(im)
    im.save("path to save the augmented image"+str(i)+".jpg")
    i+=1






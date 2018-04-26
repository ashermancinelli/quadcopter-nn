


import os
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

img_generator = ImageDataGenerator(
	rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
)


# for prefix in ['left', 'right', 'straight']:
prefix = 'straight'
for image in tqdm(os.listdir('images/' + prefix)):
	img = load_img('images/' + prefix + '/' + image)
	x = img_to_array(img)
	x = x.reshape((1,) + x.shape)
	i = 0
	for batch in img_generator.flow(x, batch_size=1,
			save_to_dir='images/'+prefix+'/expanded', save_prefix='expanded', save_format='jpeg'):
		i += 1
		if i > 10:
			break	


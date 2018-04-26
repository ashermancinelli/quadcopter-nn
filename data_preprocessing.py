
#!C:\ProgramData\Miniconda3\python.exe

from PIL import Image
import os

original_img_dir = 'original_imgs'
resized_img_dir = 'resized_imgs'
new_size = 32, 32

# Takes all images in the folder with original images 
# and resizes them to be `new_size` and saves them 
# under the same name in the folder for resized images
for f in os.listdir(original_img_dir):
    im = Image.open(original_img_dir + '/' + f)
    im = im.resize(new_size, Image.ANTIALIAS)
    im.save(resized_img_dir + '/' + f[0:-4] + '_resized.jpg')


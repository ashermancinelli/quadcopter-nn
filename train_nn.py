

import os, PIL, pickle

import tflearn
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# the images will be named either 'l', 'r', or 's' for
# 'left', 'right', or 'straight', corresponding to its label.
train_imgs = os.listdir('train_imgs')
train_labels = [ name[0] for name in train_imgs ]

train_imgs, train_labels = shuffle(train_imgs, train_labels)

train_labels = to_categorical(train_labels)

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation()

nn = input_data(
    shape=[ None, 32, 32, 3 ],
    data_preprocessing=img_prep,
    data_augmentation=img_aug
)

nn = conv_2d(nn, 32, 3, activation='relu')
nn = max_pool_2d(nn, 2)
nn = conv_2d(nn, 64, 3, activation='relu')
nn = conv_2d(nn, 64, 3, activation='relu')
nn = max_pool_2d(nn, 2)
nn = fully_connected(nn, 512, activation='relu')
nn = dropout(nn, 0.5)
nn = fully_connected(nn, 3, activation='softmax')
nn = regression(
    nn, 
    optimizer='adam',
    loss='categorical_crossentropy',
    learning_rate=1e-3
)

model = tflearn.DNN(nn, tensorboard_verbose=0)
model.fit(
    train_imgs,
    train_labels,
    n_epoch=50,
    validation_set=(test_imgs, test_labels),
    show_metric=True,
    batch_size=50,
    run_id='Parrot NN'
)
pickle.dump(model, open('model.pkl', 'wb'))


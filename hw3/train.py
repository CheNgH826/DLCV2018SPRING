from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Reshape, Dense, Dropout, Flatten
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, History, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle as pk
import sys

lable_num = 7
# abs_path = './drive/colab/DLCV2018SPRING/hw3/'
abs_path = ''

img_input = Input(shape=(512, 512, 3))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

model = Model(img_input, x)
weights_path = abs_path + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model.load_weights(weights_path, by_name=True)
# model.summary()
x = Conv2D(4096, (7,7), activation='relu', padding='same', dilation_rate=(2,2), name='fc1')(x)
x = Dropout(0.5)(x)
x = Conv2D(4096, (1,1), activation='relu', padding='same', name='fc2')(x)
x = Dropout(0.5)(x)
x = Conv2D(lable_num, (1,1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1,1), name='upsample')(x)

x = Conv2DTranspose(lable_num, (1,1), strides=(32, 32), name='deconv')(x)
x = Reshape((512*512, lable_num))(x)
# x = Flatten()(x)
x = Activation('softmax')(x)
# x = Flatten()(x)

# iter_i = int(sys.argv[1])
model = Model(img_input, x)
# if iter_i != 0:
    # model = load_model('model/iter{}.h5'.format(iter_i-1))

model.summary()

# exit(-1)

model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

val_sat = np.load(abs_path+'npy/val_sat_uint8.npy')
val_label = np.load(abs_path+'npy/val_masks_uint8.npy').reshape(-1, 512*512, lable_num)
print(val_label.shape)
print('val data loaded!')
train_sat = np.load(abs_path+'npy/train_sat_uint8.npy')
train_label = np.load(abs_path+'npy/train_masks_uint8.npy').reshape(-1, 512*512, lable_num)
print(train_label.shape)
print('train data loaded!')

###### only for testing
# train_sat = np.load('npy/mini_train_sat.npy')[:2]
# train_label = np.load('npy/mini_train_masks.npy')[:2]
# print('train data loaded!')
######

checkpointer = ModelCheckpoint(filepath=abs_path+'model/uint8.h5', verbose=0, save_best_only=True)
history = History()
earlystopping = EarlyStopping(patience=1)
model.fit(train_sat, train_label, batch_size=16, epochs=100, verbose=1, 
          validation_data=(val_sat, val_label),
          callbacks=[checkpointer, history, earlystopping])
pk.dump(history.history['loss'], open(abs_path+'model/history_train_loss.pickle', 'wb'))
pk.dump(history.history['val_loss'], open(abs_path+'model/history_val_loss.pickle', 'wb'))
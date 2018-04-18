from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Reshape, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, History, EarlyStopping
import numpy as np
import pickle as pk

lable_num = 7

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

x = Conv2DTranspose(512, (3,3), strides=(32, 32), name='deconv')(x)
x = Dense(lable_num, activation='softmax', name='softmax')(x)

model = Model(img_input, x)
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model.load_weights(weights_path, by_name=True)
model.summary()

model.compile('adam', 'categorical_crossentropy')

train_sat = np.load('npy/train_sat.npy')
train_label = np.load('npy/train_masks.npy')
print('train data loaded!')
val_sat = np.load('npy/val_sat.npy')
val_label = np.load('npy/val_masks.npy')
print('val data loaded!')

###### only for testing
# train_sat = np.load('npy/mini_train_sat.npy')[:2]
# train_label = np.load('npy/mini_train_masks.npy')[:2]
# print('train data loaded!')
######

checkpointer = ModelCheckpoint(filepath='model/test.h5', verbose=0, save_best_only=True)
history = History()
earlystopping = EarlyStopping(patience=1)
model.fit(train_sat, train_label, batch_size=128, epochs=100, verbose=1, 
          validation_data=(val_sat, val_label),
          callbacks=[checkpointer, history, earlystopping])
# print(history.history.keys())
pk.dump(history.history['loss'], open('model/history_train_loss', 'wb'))
pk.dump(history.history['val_loss'], open('model/history_val_loss', 'wb'))
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Reshape, Dense, Dropout, Flatten, UpSampling2D, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, History, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import pickle as pk
import sys

lable_num = 7
# abs_path = './drive/colab/DLCV2018SPRING/hw3/'
abs_path = ''

img_input = Input(shape=(512, 512, 3))
trainable_or_not = False
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable_or_not)(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable_or_not)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable_or_not)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable_or_not)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable_or_not)(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable_or_not)(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable_or_not)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable_or_not)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable_or_not)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable_or_not)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable_or_not)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable_or_not)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable_or_not)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

vgg_model = Model(img_input, x)
weights_path = abs_path + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
vgg_model.load_weights(weights_path, by_name=True)
# x = BatchNormalization()(x)
# vgg_model.trainable = False
x = Conv2D(4096, (7,7), activation='relu', padding='same', dilation_rate=(2,2), name='fc1')(x)
x = Dropout(0.5)(x)
x = Conv2D(4096, (1,1), activation='relu', padding='same', name='fc2')(x)
x = Dropout(0.5)(x)
x = Conv2D(lable_num, (1,1), activation='softmax', padding='valid', kernel_initializer='he_normal')(x)
x = UpSampling2D((32,32))(x)

# x = Conv2DTranspose(lable_num, (1,1), strides=(32, 32))(x)
# x = Reshape((512*512, lable_num))(x)
# x = Activation('softmax')(x)
model = Model(img_input, x)
# weights_path = abs_path + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
# model.load_weights(weights_path, by_name=True)

# iter_i = int(sys.argv[1])
# if iter_i != 0:
# model = load_model('model/uint8.h5')

model.summary()

# exit(-1)
optimizer = SGD(momentum=0.0)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

val_sat = np.load(abs_path+'npy/val_sat_uint8.npy')
val_label = np.load(abs_path+'npy/val_masks_uint8.npy')#.reshape(-1, 512*512, lable_num)
print(val_label.shape)
print('val data loaded!')
train_sat = np.load(abs_path+'npy/train_sat_uint8.npy')
train_label = np.load(abs_path+'npy/train_masks_uint8.npy')#.reshape(-1, 512*512, lable_num)
print(train_label.shape)
print('train data loaded!')

###### only for testing
# train_sat = np.load('npy/mini_train_sat.npy')[:2]
# train_label = np.load('npy/mini_train_masks.npy')[:2]
# print('train data loaded!')
######

mode = 'vggfixed'
checkpointer = ModelCheckpoint(filepath=abs_path+'model/'+mode+'_model-{epoch:02d}-{val_loss:.2f}.h5', verbose=0, save_best_only=True, period=2)
history = History()
earlystopping = EarlyStopping(patience=3, min_delta=0.01)
model.fit(train_sat, train_label, batch_size=32, epochs=20, verbose=1, 
          validation_data=(val_sat, val_label),
          callbacks=[checkpointer, history, earlystopping])
model.save('model/'+mode+'_model_final.h5')
pk.dump(history.history['loss'], open(abs_path+'model/history_train_loss.pickle', 'wb'))
pk.dump(history.history['val_loss'], open(abs_path+'model/history_val_loss.pickle', 'wb'))

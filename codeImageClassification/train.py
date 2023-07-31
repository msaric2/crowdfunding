import cv2
import tensorflow as tf
from keras.utils import Sequence
from keras import applications
from keras.layers import Dense,Flatten,Dropout,GlobalAveragePooling2D
from keras.models import Sequential,Model,load_model
from keras.optimizers import SGD
#from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import preprocess_input
import os
import os.path as osp
import numpy as np
import glob
import random
from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import auc,precision_score,recall_score,accuracy_score
from sklearn.metrics import roc_curve

FOLDER_SUCCESS_IMAGES="e:\\work\\clanci\\2021\\crowdfunding\\images\\train\\positive\\"
FOLDER_FAILED_IMAGES="e:\\work\\clanci\\2021\\crowdfunding\\images\\train\\negative\\"
FOLDER_TEST_SUCCESS_IMAGES="e:\\work\\clanci\\2021\\crowdfunding\\images\\test\\positive\\"
FOLDER_TEST_FAILED_IMAGES="e:\\work\\clanci\\2021\\crowdfunding\\images\\test\\negative\\"

def model_VGG( input_height=256, input_width=256):
    vgg_model = applications.vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(256,256,3))
   # vgg_model=applications.ResNet50(weights='imagenet',include_top=False,input_shape=(256,256,3))
    vgg_model.trainable=False
    x=vgg_model.output
    x=Flatten()(x)
    x=Dense(4096)(x)
    x=Dropout(0.2)(x)
    x = Dense(4096)(x)
    x = Dropout(0.2)(x)
    predictions=Dense(2, activation='softmax')(x)
    model = Model(inputs=vgg_model.input, outputs=predictions)
    return model


def model_VGG_Dropout_top( input_height=256, input_width=256):
    vgg_model = applications.vgg16.VGG16(weights='imagenet',include_top=True,input_shape=(256,256,3))
   # vgg_model=applications.ResNet50(weights='imagenet',include_top=False,input_shape=(256,256,3))
    vgg_model.trainable=True
    vgg_model_sequential = Sequential()
    for layer in vgg_model.layers:
        vgg_model_sequential.add(layer)
    vgg_model_sequential.add(Dense(2, activation='softmax'))
    return vgg_model_sequential




def model_Resnet( input_height=256, input_width=256):
    nClasses=2
    model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model.trainable=False
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x=Dense(4096)(x)
    x=Dropout(0.2)(x)
    x = Dense(4096)(x)
    x = Dropout(0.2)(x)
    predictions = Dense(nClasses, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    return model

def model_Densenet( input_height=256, input_width=256):
    nClasses=2
    #img_input = Input(shape=(input_height, input_width, 3))
    model = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model.trainable=False
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x=Dense(4096)(x)
    x=Dropout(0.2)(x)
    x = Dense(4096)(x)
    x = Dropout(0.2)(x)
    #x = Dense(4096)(x)
    predictions = Dense(nClasses, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    return model

class MY_Generator(Sequence):

    def __init__(self, img_paths, labels, batch_size,mean,augmentations):
        #self.image_filenames, self.labels = image_filenames, labels
        #self.batch_size,self.mean = batch_size,mean
        self.img_paths=img_paths
        self.labels=labels
        self.batch_size=batch_size
        self.mean=mean
        self.img_size=(256,256)
        self.augment=augmentations


    def __len__(self):
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        nClasses=2
        batch_x = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + (2,), dtype="uint8")

        for j,file_name in enumerate(batch_x):
            try:
                img=cv2.imread(file_name)#/255
            except Exception as e:
                print(e)
                continue
            img2=cv2.resize(img, (256, 256))
            x[j] = np.float32(img2)

        for j, v in enumerate(batch_y):
            y[j]=v
        if(self.augment!=None):
            for i in range(self.batch_size):
                sampleAugmented=self.augment(image=x[i], mask=y[i])
                x[i]=sampleAugmented["image"]
                y[i]=sampleAugmented["mask"]
        x=preprocess_input(x)
        return x,y


def train():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    test = False
    if (test):
        image_paths_test_success = glob.glob(osp.join(FOLDER_TEST_SUCCESS_IMAGES, '*.png'))
        image_paths_test_failed = glob.glob(osp.join(FOLDER_TEST_FAILED_IMAGES, '*.png'))
        image_paths_test = image_paths_test_success + image_paths_test_failed
        labels_test_success = []
        labels_test_failed = []
        for i, f in enumerate(image_paths_test_success):
            labels_test_success.append([1, 0])
        for i, f in enumerate(image_paths_test_failed):
            labels_test_failed.append([0, 1])
        labels_test = labels_test_success + labels_test_failed

        batch_size=8
        upper_limit = int(image_paths_test.__len__() / batch_size * batch_size)
        random.Random(1337).shuffle(image_paths_test)
        random.Random(1337).shuffle(labels_test)
        test_generator = MY_Generator(image_paths_test[:upper_limit], labels_test[:upper_limit], batch_size, 0, None)   #image_paths_test[10000:15000]
        model = load_model('e:\\work\\clanci\\2021\\crowdfunding\\models\\modelV11.hdf5')
        #loss, acc = model.evaluate_generator(test_generator)
        preds = model.predict_generator(test_generator)
        labs=[]
        labs = [1 if el == [1, 0] else 0 for el in labels_test]
        preds2 = [1 if el[0] > 0.5 else 0 for el in preds]
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(labs[:upper_limit], preds[:upper_limit,0])
        auc_keras = auc(fpr_keras, tpr_keras)

        #calculate recall and precision
        recall=recall_score(labs, preds2[:upper_limit])
        precision=precision_score(labs,preds2[:upper_limit])
        accuracy_score(labs, preds2[:upper_limit])


    #  input images
    image_paths_success = glob.glob(osp.join(FOLDER_SUCCESS_IMAGES, '*.png'))
    image_paths_failed = glob.glob(osp.join(FOLDER_FAILED_IMAGES, '*.png'))

    #take subset of images to speed up training
    image_paths_success_subset=image_paths_success[0:image_paths_success.__len__():3]
    image_paths_failed_subset = image_paths_failed[0:image_paths_failed.__len__():3]
    image_paths_subset=image_paths_success_subset+image_paths_failed_subset
    #generate labels for success [1,0] and failed [0,1]
    labels_success=[]
    labels_failed=[]
    for i,f in enumerate(image_paths_success_subset):
        labels_success.append([1,0])
    for i, f in enumerate(image_paths_failed_subset):
        labels_failed.append([0, 1])
    labels=labels_success+labels_failed

    #shuffle image paths and labels
    random.Random(1337).shuffle(image_paths_subset)
    random.Random(1337).shuffle(labels)
    n_val_samples=5000
    n_train_samples=image_paths_subset.__len__()-5000
    train_image_paths = image_paths_subset[:n_train_samples]
    train_labels=labels[:n_train_samples]
    val_image_paths = image_paths_subset[-n_val_samples:]
    val_labels = labels[-n_val_samples:]

    batch_size = 8
    train_generator = MY_Generator(train_image_paths, train_labels, batch_size, 0, None)
    val_generator = MY_Generator(val_image_paths, val_labels, batch_size, 0, None)

    #build model
    model=model_Densenet()
    opt = SGD(lr=1e-4)   #, momentum=0.9, nesterov=True, decay=1e-2 / 100)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    #define callbacks
    csv_logger = CSVLogger('logs/v14/log.csv', append=False, separator=';')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=1)
    rop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=2, mode='min')
    model_checkpoint = ModelCheckpoint('e:\\work\\clanci\\2021\\crowdfunding\\models\\modelV14.hdf5', monitor='val_loss')

    history_callback = model.fit_generator(train_generator,
                                           steps_per_epoch=train_generator.__len__(),
                                           nb_epoch=20,  # nb_epoch = 192,
                                           validation_data=val_generator,
                                           validation_steps=val_generator.__len__(),
                                           callbacks=[model_checkpoint, csv_logger,
                                                      TensorBoard(log_dir='./logs/v14', write_images=1, write_graph=1),
                                                      es, rop])

if __name__ == '__main__':
    train()
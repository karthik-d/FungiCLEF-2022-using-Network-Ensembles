from efficientnet.efficientnet.model import EfficientNetB6

import tensorflow as tf
import keras.utils
import numpy as np
import pandas as pd
import os
import csv
import time

IMG_SIZE=(224,224)
check=[]

#from tensorflow.keras.applications import EfficientNetB0
from keras.utils.vis_utils import plot_model

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

print(keras.__version__)
print(tf.__version__)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from keras import backend as K

#model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
model = EfficientNetB6(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
       )
# model.summary(line_length=150)

flatten = Flatten()
new_layer2 = Dense(1604, activation='softmax', name='my_dense_2')

inp2 = model.input
out2 = new_layer2(flatten(model.output))

opt = keras.optimizers.Adam(learning_rate=1e-05)

model2 = Model(inp2, out2)
model2.summary()
model2.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

#work from here

model2.load_weights('weights/weights-efficientnetb6/weights-epoch-2_001.h5')

'''modelPredictionsArray = []
outputDF = pd.DataFrame(modelPredictionsArray, columns = ['ObservationId', 'class_id'])
outputDF.to_csv('/kaggle/working/predictions.csv', index=False) # making an empty csv file

#iterating through array'''

#download and change


op_file = open('effb6predictions_2_001.csv', mode='w')
op_writer = csv.writer(op_file, delimiter=',', quotechar='"')
op_writer.writerow(['ObservationId', 'ClassId'])  
skipping = 0
i = 0

filename = "/home/miruna/LifeCLEF/FungiCLEF/FungiCLEF2022_test_metadata.csv"
df=pd.read_csv(filename)

os.chdir('Datasets/DF21-images-300/DF21_300')
time_ = time.time()
"""
for observation_id in df.ObservationId.unique():
    i += 1
    all_imgs = df.loc[df['ObservationId']==observation_id, ['filename']]
    num_rows = len(all_imgs)
    predict_probs = np.zeros((1604,))
    for idx, row in all_imgs.iterrows():
        path = row['filename']
        try:
            image = load_img(path, target_size=(224, 224))
            image = img_to_array(image)
        except OSError:
            skipping += 1
            print(skipping," skipping ", path)
            continue
        images = np.expand_dims(image, axis=0)
        predict_probs += model2.predict(images)[0]
    predicted_class = np.argmax(predict_probs)
    op_writer.writerow([observation_id, predicted_class])
    print(observation_id)
    if(i%10==0):
        print(i, '/', len(df))
        print([observation_id, predicted_class])
        #outputDF = pd.DataFrame(modelPredictionsArray, columns = ['ObservationId', 'class_id'])
        #outputDF.to_csv('/kaggle/working/outputdata.csv', index=False, mode='a', header=False)
        #print(outputDF)
        #modelPredictionsArray = []
        print("SAVING DATA")
        print(time.time()-time_)
        time_ = time.time()
op_file.close()
"""

while not df.empty:
    i += 1
    observation_id = df.iloc[0]['ObservationId']
    all_imgs = df.loc[df['ObservationId']==observation_id, ['filename']]
    num_rows = len(all_imgs)
    predict_probs = np.zeros((1604,))
    for idx, row in all_imgs.iterrows():
        path = row['filename']
        try:
            image = load_img(path, target_size=(224, 224))
            image = img_to_array(image)
        except OSError:
            skipping += 1
            print(skipping," skipping ", path)
            continue
        images = np.expand_dims(image, axis=0)
        predict_probs += model2.predict(images)[0]
    predicted_class = np.argmax(predict_probs)
    op_writer.writerow([observation_id, predicted_class])
    df.drop(df[df['ObservationId']==observation_id].index, inplace = True)
    if(i%50==0):
        print(num_rows, '/', len(df))
        print([observation_id, predicted_class])
        #outputDF = pd.DataFrame(modelPredictionsArray, columns = ['ObservationId', 'class_id'])
        #outputDF.to_csv('/kaggle/working/outputdata.csv', index=False, mode='a', header=False)
        #print(outputDF)
        #modelPredictionsArray = []
        print("SAVING DATA")
        print(time.time()-time_)
        time_ = time.time()
op_file.close()


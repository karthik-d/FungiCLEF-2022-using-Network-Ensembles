from efficientnet.efficientnet.model import EfficientNetB4

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
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate
from keras import backend as K

from sklearn.preprocessing import LabelEncoder

meta_cols = [ 'countryCode', 'level1Name', 'level2Name', 'locality', 'Substrate', 'Habitat']

#model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
model = EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
       )
# model.summary(line_length=150)

flatten = Flatten()
new_layer2 = Dense(1604, activation='softmax', name='my_dense_2')

meta_in = Input(shape=(len(meta_cols),))
img_in = model.input

out = new_layer2(Concatenate(axis=-1)([flatten(model.output), meta_in]))

model2 = Model((img_in, meta_in), out)
model2.summary()

#work from here
model2.load_weights('weights/weights-efficientnetb4-wmeta/weights-epoch-1_003.h5')

# RUN PREDICTIONS

op_file = open('effb4predictions_wmeta_1_003.csv', mode='w')
op_writer = csv.writer(op_file, delimiter=',', quotechar='"')
op_writer.writerow(['ObservationId', 'ClassId', 'Class_top3', 'Confidence_top3'])  
skipping = 0
i = 0

filename = "/home/miruna/LifeCLEF/FungiCLEF/FungiCLEF2022_test_metadata.csv"
df=pd.read_csv(filename)

"""
#MAPPING (Train --> Test)
    countryCode --> countryCode
    level1Name --> Location_lvl2
    level2Name --> Location_lvl1
    locality --> Location_lvl0
"""

meta_cols_order = [ 'countryCode', 'level1Name', 'level2Name', 'locality', 'Substrate', 'Habitat' ]
meta_cols = dict(
    countryCode = 'countryCode',
    level1Name = 'Location_lvl2',
    level2Name = 'Location_lvl1',
    locality = 'Location_lvl0',
    Substrate = 'Substrate',
    Habitat = 'Habitat'
)

meta_encoders = []
label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metalabels')
for col in meta_cols_order:
			encoder = LabelEncoder()
			encoder.classes_ = np.load(os.path.join(label_path, col+'_classes.npy'), allow_pickle=True)
			meta_encoders.append(encoder)	

os.chdir('Datasets/DF21-images-300/DF21_300')
time_ = time.time()

unseen_errors = 0
while not df.empty:

    i += 1
    observation_id = df.iloc[0]['ObservationId']
    all_imgs = df.loc[df['ObservationId']==observation_id, :]
    num_rows = len(all_imgs)
    predict_probs = np.zeros((1604,))
    instance_cnt = 0

    # Load each obs_id into a batch
    batch_meta_encoded = []
    batch_images = []
    for idx, row in all_imgs.iterrows():
        #Read metadata
        meta_encoded = []
        for encoder, col in zip(meta_encoders, meta_cols_order):
            try:
                meta_encoded.append(encoder.transform([row[meta_cols[col]]]))	
            except ValueError:
                meta_encoded.append(np.array([0])) # Unseen
                unseen_errors += 1
        batch_meta_encoded.append(np.squeeze(np.array(meta_encoded)))
        # Read image
        path = row['filename']
        try:
            image = load_img(path, target_size=(224, 224))
            image = img_to_array(image)
        except OSError:
            skipping += 1
            print(skipping," skipping ", path)
            continue
        batch_images.append(image)
        instance_cnt += 1
    
    # Make prediction in batches
    predict_probs = np.sum(model2.predict([batch_images, batch_meta_encoded]), axis=0)/instance_cnt    
    predicted_class = np.argmax(predict_probs)
    
    # Store in file
    top_3_idx = np.argpartition(predict_probs, -3)[-3:]
    op_writer.writerow([observation_id, predicted_class, top_3_idx, predict_probs[top_3_idx]])
    df.drop(df[df['ObservationId']==observation_id].index, inplace = True)

    # Verbosity
    if(i%50==0):
        print(num_rows, '/', len(df))
        print([observation_id, predicted_class])
        print("SAVING DATA")
        print(time.time()-time_)
        time_ = time.time()
        break

op_file.close()
print("Unseen Errors:", unseen_errors)

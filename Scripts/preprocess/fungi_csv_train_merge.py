import pandas as pd
import numpy as np
import os

# Numeric encode - code, endemic (0/1)
cols_x = ['observation_id', 'endemic', 'code', 'file_path']
cols_y = ['class_id']

df_train = pd.read_csv('DF20-train_metadata.csv')
#df_test = pd.read_csv('SnakeCLEF2022-TestMetadata.csv')

print(df_train.info())


#-----------------------------
# CHANGE FILES
print("Reading 1...")
train_feats_resnet = pd.read_csv('/home/miruna/LifeCLEF/EnsembleFeatures/fungi-effnet-train-features-wnew-004.csv')
#print("Merging 1...")
#df_train = df_train.merge(train_feats_resnet, how='inner', on='observation_id') 
print("Reading 2...")
train_feats_effnet = pd.read_csv('/home/miruna/LifeCLEF/EnsembleFeatures/fungi-resnet-train-features-wnew-009.csv')
#print ("Merging 2...")
#df_train = df_train.merge(train_feats_effnet, how='inner', on='observation_id')

dfs = [df_train, train_feats_resnet, train_feats_effnet]
dfs = [df.set_index('observation_id') for df in dfs]
print("Merging...")
dfs_final = dfs[0].join(dfs[1:])

#test_feats_resnet = pd.read_csv('test_img_features_inter.csv')
#test_feats_effnet = pd.read_csv('test_img_features_inter.csv')
#-------------------------------
"""
# Merging 
print(df_train.columns)
print(len(df_train.columns))
"""
print("Saving...")
#df_train.to_csv('/home/miruna/LifeCLEF/EnsembleFeatures/fungi-features-merged-train.csv')
dfs_final.to_csv('/home/miruna/LifeCLEF/EnsembleFeatures/fungi-features-merged-train.csv')

#df_test = df_test.merge(test_feats_resnet, how='inner', on='observation_id') 
#df_test = df_test.merge(test_feats_effnet, how='inner', on='observation_id') 

# print(df_train_full.columns)
# print(df_test_full.columns)

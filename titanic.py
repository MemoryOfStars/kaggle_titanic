# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:33:01 2020

@author: 75100
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd


def data_input(path):
    with open(path)as f:
        f_csv = pd.read_csv(path)
    return f_csv

'''

'''
#One-hot encoding for Embarkation
def one_hot_embarked(labels):
    result_labels = []
    for i in labels:
        if i == 'S':
            result_labels.append(np.array([1,0,0]))
        if i == 'C':
            result_labels.append(np.array([0,1,0]))
        if i == 'Q':
            result_labels.append(np.array([0,0,1]))
    return result_labels


#column 2
def process_name(row):
    pass

#column 3
def process_gender(row):
    if row[4] == 'male':
        row[4] = 0
    elif row[4] == 'female':
        row[4] = 1
    return row
        
#column 7
def process_ticket(row):
    pass

#column 9
def process_cabin(row):
    pass

#column 10
def process_embarkation(row):
    pass

def data_processing(input_data):
    train_data = []
    train_label = []
    
    '''
    Preprocessing:
        replace NaN values to a specific value
    '''
    #displace nan ages with average age
    input_data[:,5][pd.isna(input_data[:,5])] = np.mean(input_data[:,5][~pd.isna(input_data[:,5])])
    #Embarked /displaced by mode ---'S'
    input_data[:,11][pd.isna(input_data[:,11])] = 'S'
    #input_data[:,11] = one_hot_embarked(input_data[:,11])
    for row in input_data:
        new_row = []
        new_row.append(int(row[0])) #PassengerID
        new_row.append(int(row[2])) #PClass
        new_row.append(row[3])      #Name
        if row[4] == 'male':
            new_row.append(0)
        elif row[4] == 'female':
            new_row.append(1)       #Gender
        new_row.append(row[5])      #Age
        new_row.append(int(row[6])) #SibSP
        new_row.append(int(row[7])) #Parch
        new_row.append(row[8])      #Ticket
        new_row.append(int(row[9])) #Fare
        new_row.append(row[10])     #Cabin
        new_row.append(row[11])     #Embarked
        
        train_data.append(new_row)
        train_label.append(row[1])
        
    return train_data, train_label


def titanic_deepFM(train_data, train_labels, test_data, test_labels):
    
    input1 = keras.layers.Input(shape=(16,))
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(32,))
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    # 相当于 subtracted = keras.layers.subtract([x1, x2])
    subtracted = keras.layers.Subtract()([x1, x2])
    
    out = keras.layers.Dense(4)(subtracted)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)
    
    
    
    #text_input = 
    
    text_branch.add(keras.layers.Embedding(1000, 64, input_length=text_shape))
    
    numeric_branch = keras.Sequential()
    
    # 模型将输入一个大小为 (batch, input_length) 的整数矩阵。
    # 输入中最大的整数（即词索引）不应该大于 999 （词汇表大小）
    # 现在 model.output_shape == (None, 10, 64)，其中 None 是 batch 的维度。
    
    input_array = np.random.randint(1000, size=(32, 10))
    
    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)
    assert output_array.shape == (32, 10, 64)
    
    model.save('titanic_model.h5')

def train_model(train_data, train_labels, test_data, test_labels):
    pass

if __name__ == '__main__':
    input_data = data_input('train.csv')
    np_data = np.array(input_data)
    train_data, train_label = data_processing(np_data)
    print(train_data)
    print(train_label)
    #col = data[:, 5]
    #col[pd.isna(col)] = np.mean(col[~pd.isna(col)])
    #print(col)
    #print(data[np.isnan(data[:,5])])
    #print(np.mean(data[~np.isnan(data[:,5]),5]))
    #print(data.shape)
    
    #processed_data = data_processing(input_data)
    #processed_data = data_processing(input_data)
    
    #train_model()
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:03:17 2018

@author: saini
"""

from __future__ import absolute_import, division, print_function

import pathlib

import pandas as pd
import seaborn as sns

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.metrics import r2_score


column_names = ['Cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age(day)','Concrete compressive strength']

raw_dataset = pd.read_excel("Concrete_Data.xls",names=column_names,sheet_name=0,na_values = "?")

dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)




train_stats = train_dataset.describe()
train_stats.pop("Concrete compressive strength")
train_stats = train_stats.transpose()



train_labels = train_dataset.pop('Concrete compressive strength')
test_labels = test_dataset.pop('Concrete compressive strength')


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
  model = keras.Sequential([
    layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=[len(train_dataset.keys())],),
    layers.Dense(64,  kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu),
    layers.Dense(1)
  ])



  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

#model.summary()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('...............', end='')

EPOCHS = 1000




model = build_model()
test_predictions = model.predict(normed_test_data).flatten()
score=r2_score(test_labels, test_predictions, sample_weight=None, multioutput='uniform_average')


print("R^2 Score of the model is {:5.2f} ".format(score))


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)
test_predictions = model.predict(normed_test_data).flatten()
score=r2_score(test_labels, test_predictions, sample_weight=None, multioutput='uniform_average')


print("R^2 Score of the model after picking up the Weight is {:5.2f} \n\n".format(score))



def make_predictions():
   print("Please enter the following values \n")
   print("\n----------------------------------------------------------------")
   print("Cement (kg in a m^3 mixture) ")
   p1 = input()
   print("\n----------------------------------------------------------------")
   print("Blast Furnace Slag (kg in a m^3 mixture) ")
   p2 = input()
   print("\n----------------------------------------------------------------")   
   print("Fly Ash (kg in a m^3 mixture) ")
   p3 = input()
   print("\n----------------------------------------------------------------")
   print("Water (kg in a m^3 mixture) ")
   p4 = input()
   print("\n----------------------------------------------------------------")
   print("Superplasticizer (kg in a m^3 mixture) ")
   p5 = input()
   print("\n----------------------------------------------------------------")
   print("Coarse Aggregate (kg in a m^3 mixture) ")
   p6 = input()
   print("\n----------------------------------------------------------------")
   print("Fine Aggregate (kg in a m^3 mixture) ")
   p7 = input()
   print("\n----------------------------------------------------------------")
   print("Age (day) ")
   p8 = input()
   print("\n----------------------------------------------------------------")
   data = [[p1,p2,p3,p4,p5,p6,p7,p8]]
   df = pd.DataFrame(data,columns = ['Cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age(day)'],dtype=float)
   normed_df = norm(df)
   test_prediction = model.predict(normed_df).flatten()
   return test_prediction
   


choice = "y"
while choice=="y" or choice=="Y":
  print("Enter the values for which you want to predict the Concrete Strength")
  test_prediction=make_predictions()
  print("The Value of Concrete Strength is ")
  print(test_prediction)
  print("\n")
  print("Want to predict more ? (y/n)")
  choice = input()
  if choice!="y" or choice!="Y":
    print("THANK YOU...!!!")    
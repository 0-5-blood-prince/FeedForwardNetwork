import os
import wandb
import numpy as np
from keras.datasets import fashion_mnist 
from keras.utils import np_utils
labels = ["Tshirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Boot"]
def load():
  (x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
  x_train, x_val = x_train[:50000], x_train[50000:]
  y_train, y_val = y_train[:50000], y_train[50000:]
  return {
      'x_train' : x_train,
      'y_train' : y_train,
      'x_val' : x_val,
      'y_val' : y_val,
      'x_test' : x_test,
      'y_test' : y_test,
  }
def log_images():
  dataset = load()
  train_images = dataset['x_train']
  train_labels = dataset['y_train']
  set_images = []
  set_labels = []
  for i in range(len(train_images)):
    if len(set_labels) == 10:
      break
    if labels[train_labels[i]] not in set_labels:
      set_images.append(train_images[i])
      set_labels.append(labels[train_labels[i]])
  wandb.log({"Examples" : [wandb.Image(img,caption = caption) for img,caption in zip(set_images, set_labels)]})
def dataset():
  def flat(X):
    a = []
    for x in X:
      a.append((np.asarray(x)).flatten())
    return np.asarray(a)
  def normalize(X):
    return np.multiply(1/255,X)
  data = load()
  X_train = normalize(flat(data['x_train']))
  X_val = normalize(flat(data['x_val']))
  Y_train = np.eye(10)[data['y_train']]
  Y_val = np.eye(10)[data['y_val']]
  ## (50k,784) (50k,10)
  return {
      'x_train' : X_train,
      'y_train' : Y_train,
      'x_val' : X_val,
      'y_val' : Y_val,
      'x_test' : normalize(data['x_test']),
      'y_test' : data['y_test'],
  }
# def flat(X):
#   a = []
#   for x in X:
#     a.append((np.asarray(x)).flatten())
#   return np.asarray(a)
############################# Checking ############################
#wandb.init(project = 'dummy_DL')
#log_images()
# data = load()
# X_train = flat(data['x_train'])
# X_val = flat(data['x_val'])
# Y_train = np.eye(10)[data['y_train']]
# Y_val = np.eye(10)[data['y_val']]
# print(X_train.shape)
# print(Y_train)
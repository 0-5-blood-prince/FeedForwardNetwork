import os
import math
import wandb
import numpy as np
from keras.datasets import fashion_mnist, mnist
from keras.utils import np_utils
labels = ["Tshirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Boot"]

def normalize(A):
  mean = np.mean(A,axis=0)
  std = np.std(A,axis=0)
  A = A -mean
  A = A/std
  return A

def mnist_data():
  (x_train,y_train),(x_test,y_test) = mnist.load_data()
  size = len(x_train)
  train_size = math.floor( (0.9)*size)
  x_train, x_val = x_train[:train_size], x_train[train_size:]
  y_train, y_val = y_train[:train_size], y_train[train_size:]
  def flat(X):
    a = []
    for x in X:
      a.append((np.asarray(x)).flatten())
    return np.asarray(a)
  def min_max_normalize(X):
    ## Min_Max Scaling
    return np.multiply(1/255,X)
  X_train = (flat(x_train))
  X_val = (flat(x_val))
  Y_train = np.eye(10)[y_train]
  Y_val = np.eye(10)[y_val]
  
  ## (50k,784) 
  return {
      'x_train' : min_max_normalize(X_train),
      'y_train' : Y_train,
      'x_val' : min_max_normalize(X_val),
      'y_val' : Y_val,
      'x_test' : min_max_normalize(flat(x_test)),
      'y_test' : y_test,
  }

def load():
  (x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
  size = len(x_train)
  train_size = math.floor( (0.9)*size)
  x_train, x_val = x_train[:train_size], x_train[train_size:]
  y_train, y_val = y_train[:train_size], y_train[train_size:]
  ### X_train (50k,28,28) Y_train (50k,1) [0,1,...9]    [1 0 0 0 0 0 ...] , [0 0 1 0 0 0...]
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
  data = load()
  X_train = (flat(data['x_train']))
  X_val = (flat(data['x_val']))
  Y_train = np.eye(10)[data['y_train']]
  Y_val = np.eye(10)[data['y_val']]
  ## (50k,784) (50k,10)
  return {
      'x_train' : normalize(X_train),
      'y_train' : Y_train,
      'x_val' : normalize(X_val),
      'y_val' : Y_val,
      'x_test' : normalize(flat(data['x_test'])),
      'y_test' : data['y_test'],
  }
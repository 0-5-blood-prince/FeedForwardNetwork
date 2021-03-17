from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
import warnings
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb
import dataset
import NeuralNet as nn
import sys

data = dataset.dataset()
X_train = data['x_train']
Y_train = data['y_train']
X_test = data['x_test']
Y_test = data['y_test']
X_val = data['x_val']
Y_val = data['y_val']
labels = ["Tshirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Boot"]
input_dim = X_train.shape[1]
num_classes = Y_train.shape[1]

wandb.login(key="866040d7d81f67025d43e7d50ecd83d54b6cf977", relogin=False)
config_defaults = {
        'epochs' : 15,
        'batch_size' : 64,
        'weight_decay' : 0,
        'learning_rate' : 1e-4,
        'activation' : 'tanh',
        'optimizer' : 'adam',
        'num_hidden_layers' : 3,
        'hidden_layer_size' : 256,
        'momentum' : 0.9,
        'weight_init' : 'Xavier'
    }
def sassy_conf(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
        y_true:    true label of the data, with shape (nsamples,)
        y_pred:    prediction of the data, with shape (nsamples,)
        filename:  filename of figure file to save
        labels:    string array, name the order of class labels in the confusion matrix.
                    use `clf.classes_` if using scikit-learn models.
                    with shape (nclass,).
        ymap:      dict: any -> string, length == nclass.
                    if not None, map the labels & ys to more understandable strings.
                    Caution: original y_true, y_pred and labels must align.
        figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    plt.figure(dpi = 100)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="YlGnBu")
    plt.ylim([nrows, -.5])
    plt.tight_layout()
    plt.show()

def Q1():
    wandb.init(project="feedforwardfashion")
    dataset.log_images()

def train():
    wandb.init(project="feedforwardfashion",config=config_defaults)
    config = wandb.config
    sizes = [input_dim]
    activations = []
    for i in range(config.num_hidden_layers):
        sizes.append(config.hidden_layer_size)
        activations.append(config.activation)
    sizes.append(num_classes)
    activations.append('soft_max')

    network = nn.NeuralNet(config.num_hidden_layers,sizes,activations,config.weight_init)
    network.fit(config.epochs, X_train, X_val , Y_train, Y_val, config.batch_size ,'cross_entropy',config.learning_rate,
     config.momentum ,config.weight_decay ,config.optimizer)

    predictions = np.argmax(network.forward(X_test), axis=1)
    Yt = []
    pred = []
    for i in range(predictions.shape[0]):
        Yt.append(labels[Y_test[i]])
        pred.append(labels[predictions[i]])
    
    # Q7_CF(Y_test, predictions)
    # sassy_conf(Yt, pred, labels)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=Y_test, preds=predictions,
                        class_names=labels)})

# train()

def Q7():
    
    sweep_config = {
        'method': 'random', #grid, random
        'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'   
        },
        'parameters': {
            'epochs': {
                # 'values': [5, 10, 15]
                'values': [10, 15]
            },
            'num_hidden_layers': {
                # 'values': [3,4,5]
                'values': [3]
            },
            'hidden_layer_size': {  
                # 'values': [32,64,128]
                'values': [128, 256]
            },
            'weight_decay': {
                # 'values': [0, 0.0005, 0.5]
                'values': [0, 0.5]
            },
            'learning_rate': {
                # 'values': [1e-3, 1e-4]
                'values': [1e-4]
            },
            'optimizer': {
                # 'values': ['sgd', 'momentum','nesterov', 'rmsprop','adam','nadam']
                'values': ['nesterov', 'rmsprop','adam']
            },
            'batch_size':{
                # 'values': [16,32,64]
                'values': [64]
            },
            'weight_init':{
                # 'values':['random','Xavier']
                'values':['Xavier']
            },
            'activation': {
                # 'values': ['tanh', 'sigmoid', 'relu']
                'values': ['tanh']
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, entity="mooizz",project="feedforwardfashion")
    wandb.agent(sweep_id, train)
def lossdiff():
    
    wandb.init(project="feedforwardfashion",config=config_defaults)
    config = wandb.config
    sizes = [input_dim]
    activations = []
    for i in range(config.num_hidden_layers):
        sizes.append(config.hidden_layer_size)
        activations.append(config.activation)
    sizes.append(num_classes)
    activations.append('soft_max')

    network = nn.NeuralNet(config.num_hidden_layers,sizes,activations,config.weight_init)
    network.fit(config.epochs, X_train, X_val , Y_train, Y_val, config.batch_size ,config.loss_fn,config.learning_rate,
     config.momentum ,config.weight_decay ,config.optimizer)
    predictions = np.argmax(network.forward(X_test), axis=1)
    Yt = []
    pred = []
    for i in range(predictions.shape[0]):
        Yt.append(labels[Y_test[i]])
        pred.append(labels[predictions[i]])
    
    # Q7_CF(Y_test, predictions)
    # sassy_conf(Yt, pred, labels)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=Y_test, preds=predictions,
                        class_names=labels)})
                        
def Q8():
    sweep_config2 = {
        'method': 'grid', #grid, random
        'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'   
        },
        'parameters': {
            'epochs': {
                # 'values': [5, 10, 15]
                'values': [10]
            },
            'num_hidden_layers': {
                # 'values': [3,4,5]
                'values': [3]
            },
            'hidden_layer_size': {  
                # 'values': [32,64,128]
                'values': [128]
            },
            'weight_decay': {
                # 'values': [0, 0.0005, 0.5]
                'values': [0]
            },
            'learning_rate': {
                # 'values': [1e-3, 1e-4]
                'values': [1e-4]
            },
            'optimizer': {
                # 'values': ['sgd', 'momentum','nesterov', 'rmsprop','adam','nadam']
                'values': ['rmsprop','adam']
            },
            'batch_size':{
                # 'values': [16,32,64]
                'values': [64]
            },
            'weight_init':{
                # 'values':['random','Xavier']
                'values':['Xavier']
            },
            'activation': {
                # 'values': ['tanh', 'sigmoid', 'relu']
                'values': ['tanh']
            },
            'loss_fn':
            {
                'values' :['cross_entropy','square_error']
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config2, entity="mooizz",project="feedforwardfashion")
    wandb.agent(sweep_id, lossdiff)
def numbers():
    wandb.init(project="feedforwardfashion",config=config_defaults)
    config = wandb.config
    sizes = [input_dim]
    activations = []
    for i in range(config.num_hidden_layers):
        sizes.append(config.hidden_layer_size)
        activations.append(config.activation)
    sizes.append(num_classes)
    activations.append('soft_max')
    number_data = dataset.mnist_data()
    
    network = nn.NeuralNet(config.num_hidden_layers,sizes,activations,config.weight_init)
    network.fit(config.epochs, number_data['x_train'], number_data['x_val'] , number_data['y_train'] ,number_data['y_val'] , config.batch_size ,'cross_entropy',config.learning_rate,
     config.momentum ,config.weight_decay ,config.optimizer)
    
    predictions = np.argmax(network.forward(number_data['x_test']), axis=1)
    Yt = []
    pred = []
    number_test = number_data['y_test']
    for i in range(predictions.shape[0]):
        Yt.append(labels[number_test[i]])
        pred.append(labels[predictions[i]])
    # wandb.log({"test_accuracy" : network.accuracy(pred,Yt)})
    # Q7_CF(Y_test, predictions)
    # sassy_conf(Yt, pred, labels)
    # wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        # y_true=Y_test, preds=predictions,
                        # class_names=labels)})
def Q10():
    sweep_config3 = {
        'method': 'grid', #grid, random
        'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'   
        },
        'parameters': {
            'epochs': {
                # 'values': [5, 10, 15]
                'values': [10]
            },
            'num_hidden_layers': {
                # 'values': [3,4,5]
                'values': [3]
            },
            'hidden_layer_size': {  
                # 'values': [32,64,128]
                'values': [128]
            },
            'weight_decay': {
                # 'values': [0, 0.0005, 0.5]
                'values': [0]
            },
            'learning_rate': {
                # 'values': [1e-3, 1e-4]
                'values': [1e-4]
            },
            'optimizer': {
                # 'values': ['sgd', 'momentum','nesterov', 'rmsprop','adam','nadam']
                'values': ['adam']
            },
            'batch_size':{
                # 'values': [16,32,64]
                'values': [64]
            },
            'weight_init':{
                # 'values':['random','Xavier']
                'values':['Xavier']
            },
            'activation': {
                'values': ['tanh', 'sigmoid', 'relu']
                # 'values': ['tanh']
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config3, entity="mooizz",project="feedforwardfashion")
    wandb.agent(sweep_id, numbers)

if __name__ == '__main__':
    L = sys.argv[1:]
    if(len(L) == 0):
        print('Give a question number as an argument. Eg: python plots.py Q7')
        exit(0)
    if(L[0] == 'Q1'):
        Q1()
    elif(L[0] == 'Q7'):
        Q7()
    elif(L[0] == 'Q8'):
        Q8()
    elif(L[0] == "Q10"):
        numbers()
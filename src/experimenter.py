import wandb
# wandb.init(name = 'test_run',
# project='testing',
# )
import dataset
import NeuralNet as nn

data = dataset.dataset()
X_train = data['x_train']
Y_train = data['y_train']
X_val = data['x_val']
Y_val = data['y_val']
input_dim = X_train.shape[1]
num_classes = Y_train.shape[1]
sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [5, 10]
        },
        'num_hidden_layers': {
            'values': [3,4,5]
        },
        'hidden_layer_size': {  
            'values': [32,64,128]
        },
        'weight_decay': {
            'values': [0, 0.0005, 0.5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['sgd', 'momentum','nesterov', 'rmsprop','adam','nadam']
        },
        'batch_size':{
            'values': [16,32,64]
        },
        'weight_init':{
            'values':['random','Xavier']
        },
        'activation': {
            'values': ['tanh', 'sigmoid', 'relu']
        }
    }
}
wandb.login(key="866040d7d81f67025d43e7d50ecd83d54b6cf977", relogin=False)
sweep_id = wandb.sweep(sweep_config, entity="mooizz",project="testingsweep")
def train():
    config_defaults = {
        'epochs' : 5,
        'batch_size' : 64,
        'weight_decay' : 0,
        'learning_rate' : 1e-3,
        'activation' : 'tanh',
        'optimizer' : 'nadam',
        'num_hidden_layers' : 3,
        'hidden_layer_size' : 128,
        'momentum' : 0.9,
        'weight_init' : 'Xavier'
    }
    wandb.init(config=config_defaults)
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
    
# network = nn.NeuralNet(3,[input_dim,128,128,128,num_classes],['sigmoid','sigmoid','sigmoid','soft_max'],'Xavier')
# decay = 0
# network.fit(5,X_train, X_val, Y_train, Y_val, 128,'cross_entropy', 1e-3, 0.9, 0, 'nadam' )
# Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
wandb.agent(sweep_id, train)
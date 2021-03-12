import wandb
wandb.init(name = 'test_run',
project='testing',
)
import dataset
import NeuralNet as nn

data = dataset.dataset()
X_train = data['x_train']
Y_train = data['y_train']
X_val = data['x_val']
Y_val = data['y_val']
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
network = nn.NeuralNet(3,[input_dim,4,4,4,output_dim],['relu','relu','relu','soft_max'])
network.fit(X_train, X_val, Y_train, Y_val, 1000,'cross_entropy', 0.001, 'gd')
import numpy as np
import dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from scipy.special import expit
import wandb
class NeuralNet:
    def __init__(self, num_hidden_layers, layer_sizes ,activations ):
        ### first element in layer_sizes is input dimension and last element is output dimension##

        self.L = num_hidden_layers + 1 # doesn't include input layer
        self.input_dim = layer_sizes[0]
        assert(len(layer_sizes) == self.L + 1) ### Input_size, hidden layer sizes and output layer size
        self.output_dim = layer_sizes[self.L]
        assert(len(activations) == self.L)
        self.layer_sizes = layer_sizes
        self.activations = activations
        ### Initializing Weights and Biases ####
        self.b = []
        for i in range(self.L):
            self.b.append( np.zeros( (self.layer_sizes[i+1],1) ) ) ## i+1 because 0 index is the Input layer
        self.W = []
        ### Xavier Initialization Ref: deeplearning.ai/ai-notes/initializing...###
        for i in range(self.L):
            a = self.layer_sizes[i]
            b = self.layer_sizes[i+1]
            self.W.append(  np.random.randn(b,a)*( np.sqrt(2/(a+b)) ) )
    #### Activation Functions ######
    def relu(self, x):
        ## works for vector
        return np.maximum(0, x, x)
    def tanh(self, x):
        ## works for vector
        return np.tanh(x)
    def sigmoid(self, x):
        return expit(x)
        ## check if this works for a vector
        # return 1/(1 + np.exp(-x))
    ### Output Activations ###
    def softmax(self, a):
        ### checking ##
        assert(a.shape[1]==10)
        e = np.exp(a)
        return e / np.sum(e, axis=1, keepdims=True)

    def activate(self, activation, x):
        if activation == 'relu':
            # print(x)
            # print(self.relu(x))
            return self.relu(x)
        elif activation == 'tanh':
            return self.tanh(x)
        elif activation == 'sigmoid':
            return self.sigmoid(x)
        elif activation == 'soft_max':
            # print(x)
            # print(self.softmax(x))
            return self.softmax(x)
    def forward(self, inputs):
        a = []
        # print("inputs",inputs.shape)
        for l in range(1,self.L+1):
            ### Layer l ###
            ### Aggregation ###
            z = None
            if l == 1:
                input_transpose = inputs.T 
                # print(self.W[l-1].shape,input_transpose.shape)
                assert(self.W[l-1].shape[1] == input_transpose.shape[0])
                z = np.matmul(self.W[l-1],input_transpose) + self.b[l-1]
            else:
                a_transpose = a[l-2].T
                # print(self.W[l-1].shape,a_transpose.shape)
                assert(self.W[l-1].shape[1] == a_transpose.shape[0])
                z = np.matmul(self.W[l-1],a_transpose) + self.b[l-1]
            
            ### Activation ###
            z = z.T
            # print(z.shape == (inputs.shape, ))
            # print(self.activations[l-1])
            a.append( self.activate( self.activations[l-1],z))
            # print("A",a[-1].shape)
        output = a[-1]
        return output
    def softmax_grad(x):
        pass
    def activation_grads(self, activation, x):
        if activation == 'relu':
            return np.greater(x,0).astype(int)
        elif activation == 'tanh':
            return 1 - np.square(self.tanh(x))
        elif activation == 'sigmoid':
            f = self.sigmoid(x)
            return f*(1-f)
        elif activation == 'softmax':
            return self.softmax_grad(x)
    def backward_prop(self, inputs, outputs):
        ## returns gradient of weights, biases
        dW = []
        db = []
        for i in range(self.L):
            a = self.layer_sizes[i]
            b = self.layer_sizes[i+1]
            dW.append(np.zeros((b,a)))
            db.append(np.zeros((b,1)))

            
        return dW,db
    def update_params(self, grads, eta, optimizer):
        dW = grads[0]
        db = grads[1]
        ### Assuming dW and db has the grads
        if optimizer == 'gd':
            for i in range(self.L):
                self.W[i] -= np.multiply(eta,dW[i])
                self.b[i] -= np.multiply(eta,db[i])

        else:
            pass
        ### Other optimzers are to be implemented


    def accuracy(self, predicted, actual):
        class_predicted = np.argmax(predicted,axis= 1)
        class_actual = np.argmax(actual,axis=1)
        accurate = 0.0
        size = predicted.shape[0]
        for i in range(size):
            if class_actual[i]== class_predicted[i]:
                accurate += 1.0
        return (accurate/size)*100


    def fit(self, train_inputs, valid_inputs, train_outputs, valid_outputs, batch_size, loss_fn, eta, optimizer):
        ## All inputs and outputs are numpy arrays
        ### Gradient Descent ####
        train_size = train_inputs.shape[0]
        valid_size = valid_inputs.shape[0] 
        t = 0
        max_epochs = 10
        while(t<max_epochs):
            t+=1
            loss = 0
            #### Minibatch ####
            st = 0
            end = batch_size
            while(end <= train_size):
            
                mini_input = train_inputs[st:end]
                mini_output = train_outputs[st:end]
                st = end 
                end = st+batch_size 
                print("batch",end/batch_size)
                ### Network predicted outputs ###
                y_hat = self.forward(mini_input)
                assert(y_hat.shape  == mini_output.shape)
                grads = self.backward_prop(mini_input,mini_output)
                ## grads should have dW and db
                self.update_params(grads , eta, optimizer)
                

                ##update loss ##
                if loss_fn == 'cross_entropy':
                    loss += log_loss(mini_output,y_hat)
                elif loss_fn == 'square_error':
                    loss += mean_squared_error(mini_output,y_hat)

            net_pred = self.forward(train_inputs)
            ## Print Loss and change in accuracy for each epoch for Training####
            train_accuracy = self.accuracy(net_pred, train_outputs)
            print("Epoch :", t,"Training Loss :",loss, "Training Accuracy :", train_accuracy )

            net_pred_valid = self.forward(valid_inputs)
            valid_loss = 0
            ## Print Loss and change in accuracy for each epoch for Validation####
            if loss_fn == 'cross_entropy':
                valid_loss += log_loss(valid_outputs,net_pred_valid)
            elif loss_fn == 'square_error':
                valid_loss += mean_squared_error(valid_outputs,net_pred_valid)
            valid_accuracy = self.accuracy(net_pred_valid, valid_outputs)
            print("Epoch :", t,"Validation Loss :",valid_loss, "Validation Accuracy :",valid_accuracy )
            wandb.log({ "Epoch": t, "Train Loss": loss, "Train Acc": train_accuracy, "Valid Loss": valid_loss, "Valid Acc": valid_accuracy})

        ### log training ............... ###
data = dataset.dataset()
X_train = data['x_train']
Y_train = data['y_train']
X_val = data['x_val']
Y_val = data['y_val']
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
nn = NeuralNet(3,[input_dim,4,4,4,output_dim],['relu','relu','relu','soft_max'])
nn.fit(X_train, X_val, Y_train, Y_val, 1000,'cross_entropy', 0.001, 'gd')
















    


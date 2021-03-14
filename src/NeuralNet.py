import numpy as np
import dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from scipy.special import expit
from scipy.special import softmax as sm
import scipy.special as sc
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
        self.aggLayer = []  # to store h's
        self.actLayer = []  # to store a's
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
    def softmax(self, x):
        return np.exp(x - sc.logsumexp(x))
    # def softmax(self, a):
        
    #     ### checking ##
        # return sm(a)
    #     # assert(a.shape[1]==10)
    #     # e = np.exp(a)
    #     # return e / np.sum(e, axis=1, keepdims=True)

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
        h = []
        self.aggLayer = []
        self.actLayer = []

        # print("inputs",inputs.shape)
        # appending inputLayer to self.actLayer
        self.actLayer.append(inputs)
        self.aggLayer.append(inputs)
        for l in range(1,self.L+1):
            ### Layer l ###
            ### Aggregation ###
            a = None
            if l == 1:
                input_transpose = inputs.T 
                # print(self.W[l-1].shape,input_transpose.shape)
                assert(self.W[l-1].shape[1] == input_transpose.shape[0])
                a = np.matmul(self.W[l-1],input_transpose) + self.b[l-1]
            else:
                h_transpose = h[l-2].T
                # print(self.W[l-1].shape,h_transpose.shape)
                assert(self.W[l-1].shape[1] == h_transpose.shape[0])
                a = np.matmul(self.W[l-1],h_transpose) + self.b[l-1]
            
            self.aggLayer.append(a[-1])

            ### Activation ###
            a = a.T
            # print(a.shape == (inputs.shape, ))
            # print(self.activations[l-1])
            h.append( self.activate( self.activations[l-1],a))
            # print("H",h[-1].shape)

            self.actLayer.append(h[-1])

        output = h[-1]
        # print('nn 94, actLayer shape:' + str(len(self.actLayer[0])))
        


        return output
    def softmax_grad(self, x):
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

    def backward_prop(self, inputs, outputs, lossFunction, W, b, activations):
        ## returns gradient of weights, biases
        num_samples = inputs.shape[0]
        output_dim = outputs.shape[1]
        
        x = inputs
        y = outputs.reshape((num_samples,output_dim))
        fx = self.actLayer[-1].reshape((num_samples,output_dim))
        # print("ACtlayer size",len(self.actLayer))
        dw = []
        db = []
        da = []
        # dh = []
        # dh.append(np.zeros((inputs.shape[1],num_samples)))
        da.append(np.zeros((inputs.shape[1],num_samples)))
        for i in range(self.L):
            prev = self.layer_sizes[i]  
            curr = self.layer_sizes[i+1]
            dw.append(np.zeros((curr,prev)))
            db.append(np.zeros((curr,1)))
            # dh.append(np.zeros((curr,num_samples)))
            da.append(np.zeros((curr,num_samples)))

   
            # cross entropy
        # fx-y(outputLay) dim (numSamp, outputDim)
        da[self.L] = (fx-y).T
        for l in range(self.L,0,-1):
            # print("Layer",l)
            # for ll in range(0, self.L+1):
            #     print('hi a h shapes', da[ll].shape, dh[ll].shape)
            # for ll in range(self.L):
            #     print('hi W b',dw[ll].shape,db[ll].shape)
            h = self.actLayer[l-1].reshape((num_samples,self.layer_sizes[l-1]))
            # print(da[l].shape,h.T.shape)
            # print(fx)
            # print(y)
            # print(l)
            dw[l-1] = (1/num_samples)*np.matmul(da[l],h)
            db[l-1] = (1/num_samples)*np.sum(da[l], axis = 1, keepdims=True)
            # print(W[l-1].T.shape, da[l].shape, l)
            dh = np.matmul(W[l-1].T, da[l])
            hadamardProd = self.activation_grads(activations, self.aggLayer[l-1].T)
            da[l-1] = np.multiply(dh,hadamardProd)

        
        return dw,db

    def accuracy(self, predicted, actual):
        class_predicted = np.argmax(predicted,axis= 1)
        class_actual = np.argmax(actual,axis=1)
        accurate = 0.0
        size = predicted.shape[0]
        for i in range(size):
            if class_actual[i]== class_predicted[i]:
                accurate += 1.0
        return (accurate/size)*100
    def SGD(self, eta, gamma, optimizer,  train_inputs, train_outputs, batch_size, loss_fn , decay):
            st = 0
            end = batch_size
            sample_size = train_inputs.shape[0]
            loss = 0
            while(end <= sample_size):
            
                mini_input = train_inputs[st:end]
                mini_output = train_outputs[st:end]
                st = end 
                end = st+batch_size 
                # print("batch",end/batch_size)
                
                ### Network predicted outputs ###
                y_hat = self.forward(mini_input)

                assert(y_hat.shape  == mini_output.shape)

                grads = self.backward_prop(mini_input,mini_output, 'cross_entropy',self.W,self.b,self.activations[0])
                ## grads should have dW and db
                
                dW = grads[0]
                db = grads[1]
                momentW = []
                momentb = []
                for i in range(self.L):
                    a = self.layer_sizes[i]
                    b = self.layer_sizes[i+1]
                    momentW.append(np.zeros((b,a)))
                    momentb.append(np.zeros((b,1)))
                ### Assuming dW and db has the grads
                if optimizer == 'sgd':
                    for i in range(self.L):
                        self.W[i] -= np.multiply(eta,dW[i])
                        self.b[i] -= np.multiply(eta,db[i])

                elif optimizer == 'momentum' or optimizer == 'nesterov':
                    for i in range(self.L):
                        momentW[i] = np.multiply(gamma,momentW[i]) + np.multiply(eta,dW[i])
                        momentb[i] = np.multiply(gamma,momentb[i])+ np.multiply(eta,db[i])
                        self.W[i] -= momentW[i]
                        self.b[i] -= momentb[i]
                # if(end == sample_size):
                    

                ##update loss ##
                if loss_fn == 'cross_entropy':
                    loss += log_loss(mini_output,y_hat)
                elif loss_fn == 'square_error':
                    loss += mean_squared_error(mini_output,y_hat)
                
            print('W', self.W)
            print('b', self.b)
            return loss

    def fit(self, train_inputs, valid_inputs, train_outputs, valid_outputs, batch_size, loss_fn, eta, gamma, decay, optimizer):
        ## All inputs and outputs are numpy arrays
        ### Gradient Descent ####
        train_size = train_inputs.shape[0]
        valid_size = valid_inputs.shape[0] 
        t = 0
        max_epochs = 100
        while(t<max_epochs):
            t+=1
            loss = 0
            #### Minibatch ####
            if optimizer == 'sgd' or optimizer == 'momentum' or optimizer == 'nesterov':
                loss = self.SGD( eta,gamma , optimizer , train_inputs , train_outputs , batch_size , loss_fn,decay)

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
            # wandb.log({ "Epoch": t, "Train Loss": loss, "Train Acc": train_accuracy, "Valid Loss": valid_loss, "Valid Acc": valid_accuracy})

        ### log training ............... ###

















    


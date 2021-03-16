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
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="YlGnBu")
    plt.ylim([nrows, -.5])
    plt.tight_layout()
    plt.show()

def Q1():
    wandb.init(project="feedforwardfashion")
    dataset.log_images()

def train():
    config_defaults = {
        'epochs' : 10,
        'batch_size' : 64,
        'weight_decay' : 0.0005,
        'learning_rate' : 1e-4,
        'activation' : 'tanh',
        'optimizer' : 'adam',
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

    predictions = np.argmax(nn.forward(X_test), axis=1)
    # Q7_CF(Y_test, predictions)
    sassy_conf(Y_test, predictions, labels)

train()

def Q7_CF(y_true, y_pred):
    
    sweep_config = {
        'method': 'bayes', #grid, random
        'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'   
        },
        'parameters': {
            'epochs': {
                # 'values': [5, 10, 15]
                'values': [5, 10, 15, 30]
            },
            'num_hidden_layers': {
                # 'values': [3,4,5]
                'values': [3,4,5]
            },
            'hidden_layer_size': {  
                # 'values': [32,64,128]
                'values': [32,64,128,256]
            },
            'weight_decay': {
                # 'values': [0, 0.0005, 0.5]
                'values': [0, 0.0005, 0.5]
            },
            'learning_rate': {
                # 'values': [1e-3, 1e-4]
                'values': [1e-3, 1e-4]
            },
            'optimizer': {
                # 'values': ['sgd', 'momentum','nesterov', 'rmsprop','adam','nadam']
                'values': ['sgd', 'momentum','nesterov', 'rmsprop','adam','nadam']
            },
            'batch_size':{
                # 'values': [16,32,64]
                'values': [16,32,64]
            },
            'weight_init':{
                # 'values':['random','Xavier']
                'values':['Xavier']
            },
            'activation': {
                # 'values': ['tanh', 'sigmoid', 'relu']
                'values': ['tanh', 'relu']
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, entity="mooizz",project="feedforwardfashion")
    wandb.agent(sweep_id, train)

    
    


# class ConfusionPlots():

# 	def pretty_conf(test,pred,st,columns=None):

# 		def get_new_fig(fn, figsize=[9,9]):
# 			""" Init graphics """
# 			fig1 = plt.figure(fn, figsize)
# 			ax1 = fig1.gca()   #Get Current Axis
# 			ax1.cla() # clear existing plot
# 			return fig1, ax1

# 		def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
# 			"""
# 			  config cell text and colors
# 			  and return text elements to add and to dell
# 			  @TODO: use fmt
# 			"""
# 			text_add = []; text_del = [];
# 			cell_val = array_df[lin][col]
# 			tot_all = array_df[-1][-1]
# 			per = (float(cell_val) / tot_all) * 100
# 			curr_column = array_df[:,col]
# 			ccl = len(curr_column)

# 			#last line  and/or last column
# 			if(col == (ccl - 1)) or (lin == (ccl - 1)):
# 				#tots and percents
# 				if(cell_val != 0):
# 					if(col == ccl - 1) and (lin == ccl - 1):
# 						tot_rig = 0
# 						for i in range(array_df.shape[0] - 1):
# 							tot_rig += array_df[i][i]
# 						per_ok = (float(tot_rig) / cell_val) * 100
# 					elif(col == ccl - 1):
# 						tot_rig = array_df[lin][lin]
# 						per_ok = (float(tot_rig) / cell_val) * 100
# 					elif(lin == ccl - 1):
# 						tot_rig = array_df[col][col]
# 						per_ok = (float(tot_rig) / cell_val) * 100
# 					per_err = 100 - per_ok
# 				else:
# 					per_ok = per_err = 0

# 				per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

# 				#text to DEL
# 				text_del.append(oText)

# 				#text to ADD
# 				font_prop = fm.FontProperties(weight='bold', size=fz)
# 				text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
# 				lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
# 				lis_kwa = [text_kwargs]
# 				dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
# 				dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
# 				lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
# 				for i in range(len(lis_txt)):
# 					newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
# 					#print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
# 					text_add.append(newText)
# 				#print '\n'

# 				#set background color for sum cells (last line and last column)
# 				carr = [0.27, 0.30, 0.27, 1.0]
# 				if(col == ccl - 1) and (lin == ccl - 1):
# 					carr = [0.17, 0.20, 0.17, 1.0]
# 				facecolors[posi] = carr

# 			else:
# 				if(per > 0):
# 					txt = '%s\n%.2f%%' %(cell_val, per)
# 				else:
# 					if(show_null_values == 0):
# 						txt = ''
# 					elif(show_null_values == 1):
# 						txt = '0'
# 					else:
# 						txt = '0\n0.0%'
# 				oText.set_text(txt)

# 				#main diagonal
# 				if(col == lin):
# 					#set color of the textin the diagonal to white
# 					oText.set_color('w')
# 					# set background color in the diagonal to blue
# 					facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
# 				else:
# 					oText.set_color('r')

# 			return text_add, text_del
# 		#

# 		def insert_totals(df_cm):
# 			""" insert total column and line (the last ones) """
# 			sum_col = []
# 			for c in df_cm.columns:
# 				sum_col.append( df_cm[c].sum() )
# 			sum_lin = []
# 			for item_line in df_cm.iterrows():
# 				sum_lin.append( item_line[1].sum() )
# 			df_cm['sum_lin'] = sum_lin
# 			sum_col.append(np.sum(sum_lin))
# 			df_cm.loc['sum_col'] = sum_col
# 			#print ('\ndf_cm:\n', df_cm, '\n\b\n')
# 		#

# 		def pretty_plot_confusion_matrix(df_cm,st,annot=True, cmap="Oranges", fmt='.2f', fz=20,
# 			  lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y'):
# 			"""
# 			  print conf matrix with default layout (like matlab)
# 			  params:
# 				df_cm          dataframe (pandas) without totals
# 				annot          print text in each cell
# 				cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
# 				fz             fontsize
# 				lw             linewidth
# 				pred_val_axis  where to show the prediction values (x or y axis)
# 								'col' or 'x': show predicted values in columns (x axis) instead lines
# 								'lin' or 'y': show predicted values in lines   (y axis)
# 			"""
# 			if(pred_val_axis in ('col', 'x')):
# 				xlbl = 'Predicted'
# 				ylbl = 'Actual'
# 			else:
# 				xlbl = 'Actual'
# 				ylbl = 'Predicted'
# 				df_cm = df_cm.T

# 			# create "Total" column
# 			insert_totals(df_cm)

# 			#this is for print allways in the same window
# 			fig, ax1 = get_new_fig('Conf matrix default', figsize)

# 			#thanks for seaborn
# 			ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
# 							cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

# 			#set ticklabels rotation
# 			ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 15)
# 			ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 15)

# 			# Turn off all the ticks
# 			for t in ax.xaxis.get_major_ticks():
# 				t.tick1On = False
# 				t.tick2On = False
# 			for t in ax.yaxis.get_major_ticks():
# 				t.tick1On = False
# 				t.tick2On = False

# 			#face colors list
# 			quadmesh = ax.findobj(QuadMesh)[0]
# 			facecolors = quadmesh.get_facecolors()

# 			#iter in text elements
# 			array_df = np.array( df_cm.to_records(index=False).tolist() )
# 			text_add = []; text_del = [];
# 			posi = -1 #from left to right, bottom to top.
# 			for t in ax.collections[0].axes.texts: #ax.texts:
# 				pos = np.array( t.get_position()) - [0.5,0.5]
# 				lin = int(pos[1]); col = int(pos[0]);
# 				posi += 1
# 				#print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

# 				#set text
# 				txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

# 				text_add.extend(txt_res[0])
# 				text_del.extend(txt_res[1])

# 			#remove the old ones
# 			for item in text_del:
# 				item.remove()
# 			#append the new ones
# 			for item in text_add:
# 				ax.text(item['x'], item['y'], item['text'], **item['kw'])

# 			#titles and legends
# 			ax.set_title('Confusion matrix for'+st)
# 			ax.set_xlabel(xlbl)
# 			ax.set_ylabel(ylbl)
# 			plt.ylim([len(df_cm), -.5])
# 			plt.tight_layout()  #set layout slim
# 			plt.show()
# 		#

# 		def plot_confusion_matrix_from_data(y_test,st,predictions, columns=None, annot=True, cmap="Oranges",
# 			  fmt='.2f', fz=20, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin'):
# 			"""
# 				plot confusion matrix function with y_test (actual values) and predictions (predic),
# 				whitout a confusion matrix yet
# 			"""
# 			from sklearn.metrics import confusion_matrix
# 			from pandas import DataFrame

# 			#data
# 			if(columns == None)[0]:
# 				#labels axis integer:
# 				##columns = range(1, len(np.unique(y_test))+1)
# 				#labels axis string:
# 				from string import ascii_uppercase
# 				columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

# 			confm = confusion_matrix(y_test, predictions)
# 			cmap = 'Oranges';
# 			fz = 20;
# 			figsize=[9,9];
# 			show_null_values = 2
# 			df_cm = DataFrame(confm, index=columns, columns=columns)
# 			pretty_plot_confusion_matrix(df_cm,st,fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)
		


# 		""" test function with y_test (actual values) and predictions (predic) """
# 		#data
# 		warnings.filterwarnings("ignore")
# 		y_test = np.array(test)
# 		predic = np.array(pred)
# 		"""
# 		  Examples to validate output (confusion matrix plot)
# 			actual: 5 and prediction 1   >>  3
# 			actual: 2 and prediction 4   >>  1
# 			actual: 3 and prediction 4   >>  10
# 		"""
# 		annot = True;
# 		cmap = 'Oranges';
# 		fmt = '.2f'
# 		lw = 0.5
# 		cbar = False
# 		show_null_values = 2
# 		pred_val_axis = 'y'
# 		#size::
# 		fz = 20;
# 		figsize = [9,9];
# 		if(len(y_test) > 10):
# 			fz=20; figsize=[14,14];
# 		plot_confusion_matrix_from_data(y_test," "+st,predic, columns,
# 		  annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


# 	def sassy_conf(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
# 		"""
# 		Generate matrix plot of confusion matrix with pretty annotations.
# 		The plot image is saved to disk.
# 		args: 
# 		  y_true:    true label of the data, with shape (nsamples,)
# 		  y_pred:    prediction of the data, with shape (nsamples,)
# 		  filename:  filename of figure file to save
# 		  labels:    string array, name the order of class labels in the confusion matrix.
# 					 use `clf.classes_` if using scikit-learn models.
# 					 with shape (nclass,).
# 		  ymap:      dict: any -> string, length == nclass.
# 					 if not None, map the labels & ys to more understandable strings.
# 					 Caution: original y_true, y_pred and labels must align.
# 		  figsize:   the size of the figure plotted.
# 		"""
# 		if ymap is not None:
# 			y_pred = [ymap[yi] for yi in y_pred]
# 			y_true = [ymap[yi] for yi in y_true]
# 			labels = [ymap[yi] for yi in labels]
# 		cm = confusion_matrix(y_true, y_pred, labels=labels)
# 		cm_sum = np.sum(cm, axis=1, keepdims=True)
# 		cm_perc = cm / cm_sum.astype(float) * 100
# 		annot = np.empty_like(cm).astype(str)
# 		nrows, ncols = cm.shape
# 		for i in range(nrows):
# 			for j in range(ncols):
# 				c = cm[i, j]
# 				p = cm_perc[i, j]
# 				if i == j:
# 					s = cm_sum[i]
# 					annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
# 				elif c == 0:
# 					annot[i, j] = ''
# 				else:
# 					annot[i, j] = '%.1f%%\n%d' % (p, c)
# 		cm = pd.DataFrame(cm, index=labels, columns=labels)
# 		cm.index.name = 'Actual'
# 		cm.columns.name = 'Predicted'
# 		fig, ax = plt.subplots(figsize=figsize)
# 		sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="YlGnBu")
# 		plt.ylim([nrows, -.5])
# 		plt.tight_layout()
# 		plt.show()

        
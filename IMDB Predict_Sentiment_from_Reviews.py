from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def get_model():
	aliases = {}
	Input_1 = Input(shape=(499,), name='Input_1')
	Embedding_1 = Embedding(name='Embedding_1',dropout= 0.2,output_dim= 128,input_dim= 5000)(Input_1)
	LSTM_1 = LSTM(name='LSTM_1',activation= 'tanh' ,output_dim= 128,dropout_U= 0.2,dropout_W= 0.2)(Embedding_1)
	Dense_2 = Dense(name='Dense_2',activation= 'sigmoid' ,output_dim= 1)(LSTM_1)

	model = Model([Input_1],[Dense_2])
	return aliases, model


from keras.optimizers import *

def get_optimizer():
	return Adam(beta_2=0.999,epsilon=1e-08,decay=0,beta_1=0.9,lr=0.001)

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'binary_crossentropy'

def get_batch_size():
	return 32

def get_num_epoch():
	return 15

def get_data_config():
	return '{"kfold": 1, "samples": {"validation": 5000, "training": 20000, "split": 1, "test": 0}, "datasetLoadOption": "batch", "shuffle": false, "numPorts": 1, "mapping": {"Sentiment": {"port": "OutputPort0", "type": "Numeric", "options": {"Normalization": false, "Scaling": 1}, "shape": ""}, "Review": {"port": "InputPort0", "type": "Array", "options": {"Normalization": false, "Scaling": 1}, "shape": ""}}, "dataset": {"type": "public", "samples": 25000, "name": "imdb"}}'

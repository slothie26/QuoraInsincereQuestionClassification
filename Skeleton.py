#Skeleton - enable word wrap
def read_dataset():
	#priority: very very high
	#read the dataset, generate embeddings for it and create a pickle out of it.
	#dont do everything within this function, modularise and create fns as required
	pass
def load_dataset():
	#prioirity: high
	#open the pickle files and load the dataset
	pass

def preprocess():
	#priority:high
	#all preprocessing functions need to be called here. Define each preprocessing step as a seperate function for easy addition and removal of functions in the future
	'''Tokenizer to be used (keras, spacy) Priority: high
	Case updation Priority: high
	Spelling correction? Priority: low
	Lemmatising? Priority: low
	Punctuation removal? Priority: high
	Spaces around characters Priority: high'''
	pass
def random_sampling():
	#priority: low
	#performs random sampling and reduces size of embeddings
	pass
def model_definition(): 
	#priority: very very high - VIDHU
	#define all the layers of the model

def model_training():
	#priority: high
	#define the training phase of the model, along with cross validation, hyperparameter tuning save the final model

def validation():
	#priority: low
	#load saved model use the validation set to measure accuracy
#add steps for gpu usage

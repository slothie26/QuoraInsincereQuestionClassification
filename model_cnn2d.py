import pickle
import os
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm

batch_size = 2048
num_files = 2
embedding_size = 300
question_size = 30
max_features = 40000
#activ = 'relu'
activ = 'tanh'
def get_concatenated_embeddings(temp_df):
#        print("tdf", temp_df.shape)
        zero_embeddings = np.zeros(embedding_size)
        truncated_df = temp_df[:question_size]
#        print(truncated_df.shape)
#        print(len(truncated_df),len(truncated_df[0]))
#        print(type([zero_embeddings]*(question_size-len(truncated_df))))
#        print(np.array([zero_embeddings]*(question_size - len(truncated_df))).shape)
        if(len(truncated_df)!=30):
            truncated_df = np.concatenate((truncated_df, np.array([zero_embeddings]*(question_size- len(truncated_df)))))
#        truncated_df.append(zero_embeddings*(question_size - len(truncated_df)))
        return truncated_df


def batch_gen(n_batches,y,all_embeddings):
#    print("bg")
    while True: 
        for i in range(n_batches):
#            print("for loop")
            embeddings_list = all_embeddings[i*batch_size:(i+1)*batch_size] 
#           print("el",embeddings_list.shape,all_embeddings.shape,embeddings_list[0].shape)
            concat_embed_ques = np.array([get_concatenated_embeddings(ques) for ques in embeddings_list])
            #print(i,concat_embed_ques.shape)
            yield concat_embed_ques, np.array(y[i*batch_size:(i+1)*batch_size])

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
all_embeddings = pickle.load(open("merged_embeddings_train","rb"))
all_embeddings_test = pickle.load(open("merged_embeddings_test","rb"))	
#print(all_embeddings.shape, all_embeddings[0].shape)
#print(type(all_embeddings),type(all_embeddings[0]))
all_embeddings_train, all_embeddings_test,all_y_train,all_y_test = train_test_split(all_embeddings, train_df["target"][0:all_embeddings.shape[0]], test_size = 0.30, random_state = 42)
embeddings_train,embeddings_val,y_train,y_val= train_test_split(all_embeddings_train,all_y_train,test_size=0.20)
n_batches = math.ceil(len(y_train)/batch_size)
print("train set",len(y_train))
print("test set",len(all_y_test))
print("val set",len(y_val))
bg = batch_gen(n_batches,y_train,embeddings_train)
val_vects = np.array([get_concatenated_embeddings(val_emb) for val_emb in embeddings_val][:3000])
val_y = np.array(y_val[:3000])



from keras.models import Sequential
from keras.layers import *

filters = 128
model = Sequential()
model.add(Conv2D(filters,kernel_size = (1,embedding_size),activation = activ, input_shape = (question_size,embedding_size)))
model.add(Conv2D(filters,kernel_size = (2, embedding_size),activation = activ))
model.add(Conv2D(filters,kernel_size = (3, embedding_size),activation = activ))
model.add(Conv2D(filters,kernel_size = (5, embedding_size),activation = activ))

model.add(GlobalAveragePooling1D())

model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(1,activation= 'sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit_generator(bg, epochs=1,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),
                    verbose=True)

#test_vects = np.array([get_concatenated_embeddings(test_emb) for test_emb in all_embeddings_test])
#test_y = np.array(y_test[:3000])
test_gen=batch_gen(n_batches,all_y_test,all_embeddings_test)
scores=model.evaluate_generator(test_gen,steps=25,verbose=1)
print("Accuracy", scores[1])

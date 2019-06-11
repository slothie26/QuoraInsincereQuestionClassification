import pickle
import os
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
<<<<<<< HEAD
from sklearn.metrics import *

=======
>>>>>>> 77a9dd4960e57bdf719df8a0d256242ad9f62448

batch_size = 128
num_files = 10
embedding_size = 300
question_size = 30

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
<<<<<<< HEAD
#            print(i,concat_embed_ques.shape)
=======
            #print(i,concat_embed_ques.shape)
>>>>>>> 77a9dd4960e57bdf719df8a0d256242ad9f62448
            yield concat_embed_ques, np.array(y[i*batch_size:(i+1)*batch_size])

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
all_embeddings = pickle.load(open("merged_embeddings_train","rb"))
#all_embeddings_test = pickle.load(open("merged_embeddings_test","rb"))	
<<<<<<< HEAD
print(all_embeddings.shape, all_embeddings[0].shape)
print(type(all_embeddings),type(all_embeddings[0]))
=======
#print(all_embeddings.shape, all_embeddings[0].shape)
#print(type(all_embeddings),type(all_embeddings[0]))
>>>>>>> 77a9dd4960e57bdf719df8a0d256242ad9f62448
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
from keras.layers import CuDNNLSTM, Dense, Bidirectional
from sklearn.metrics import f1_score
model = Sequential()
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True),
                        input_shape=(30, 300)))
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
<<<<<<< HEAD
history = model.fit_generator(bg, epochs=10,
=======
model.fit_generator(bg, epochs=2,
>>>>>>> 77a9dd4960e57bdf719df8a0d256242ad9f62448
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),
                    verbose=True)

#test_vects = np.array([get_concatenated_embeddings(test_emb) for test_emb in all_embeddings_test])
#test_y = np.array(y_test[:3000])
test_gen=batch_gen(n_batches,all_y_test,all_embeddings_test)
scores=model.evaluate_generator(test_gen,steps=400,verbose=1)
print("Accuracy", scores[1])
<<<<<<< HEAD
batch_size = 30000
test_whole = batch_gen(1, all_y_test,all_embeddings_test)
import matplotlib.pyplot as plt
pred_prob=model.predict_generator(test_whole,steps=1,verbose=1)
pred_y = pred_prob > 0.5
print("F1 score: ",f1_score(all_y_test[:batch_size],pred_y))
print("Confusion_matrix:\n",confusion_matrix(all_y_test[:batch_size],pred_y))
precision, recall, _ =  precision_recall_curve(all_y_test[:batch_size],pred_prob)
plt.title("Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')
# show the plot
plt.show()
plt.savefig('bilstm_prc.png')

plt.figure()
print("roc curve \n")
fpr, tpr, threshold =  roc_curve(all_y_test[:batch_size],pred_prob)
roc_auc =  roc_auc_score(all_y_test[:batch_size],pred_prob)


plt.title("Receiver Operating Characeristic")
plt.plot(fpr,tpr,'b', label = 'AUC = %0.2f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()
plt.savefig('bilstm_roc.png')


# summarize history for accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('bilstm_acc_history.png')

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('bilstm_loss_history.png')
=======
pred_val=model.predict_generator(test_gen, steps=400,verbose=1)
print("Predicted val", pred_val[1])
#f1_score= fi_score(
>>>>>>> 77a9dd4960e57bdf719df8a0d256242ad9f62448

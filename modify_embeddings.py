import pickle
import numpy as np

def merge_embeddings(num_files,file_name):
	ques_embed_list = []
	for i in range(num_files):
		print(i)
		current_data = pickle.load(open(file_name+str(i),"rb"))
		for questions in current_data:
			word_embed_list = []
			for words,word_embed in questions.items():
				word_embed_list.append(word_embed)
			ques_embed_list.append(word_embed_list)
	print(len(ques_embed_list),len(ques_embed_list[0]),len(ques_embed_list[0][0]))
	main_df=np.array([np.array(xi) for xi in ques_embed_list])
	return main_df

print("train")
main_df_train = merge_embeddings(4,"final_train_embeddings_")
pickle.dump(main_df_train,open("final_merged_embeddings_train_4","wb"))

print("test")
#main_df_test = merge_embeddings(4,"final_test_embeddings_")
#pickle.dump(main_df_test,open("final_merged_embeddings_test","wb"))

import sys
import time
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import operator
import pickle
from gensim.models import KeyedVectors
from collections import OrderedDict

def create_pickle(pickle_path, py_obj):
    py_obj.to_pickle(pickle_path)

def read_dataset(path):
    return pd.read_csv(path, engine = 'python')

def load_dataset(pickle_path):
    df = pd.read_pickle(pickle_path)
    return df

def build_vocab(sentences, verbose =  True):
    #sentences: list of list of words
    #return: dictionary of words and their count

    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)

    return x

def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

def _get_mispell(mispell_dict):
        mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
        return mispell_dict, mispell_re

def replace_typical_misspell(text, mispellings, mispellings_re):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

def build_embeddings():
    # The following snippet has already been run
    # The pickle file(s) of the Embedding Indices have already been created
    # We only need to read them as below

    '''
    news_path_google = "GoogleNews-vectors-negative300.bin"
    #news_path_wiki = "./Embeddings/wiki-news-300d-1M.vec"
    #paragram = "./Embeddings/paragram_300_sl999.txt"
    #glove = "./Embeddings/glove.840B.300d.txt"

    #Writing Embedding Indices
    f_google = open("emb_data_google", "wb")
    #f_wiki = open("emb_data_wiki", "wb")
    #f_glove = open("emp_data_glove", "wb")
    #f_para = open("emp_data_para", "wb")

    emb_ind_google = KeyedVectors.load_word2vec_format(news_path_google, binary=True)
    #emb_ind_wiki = KeyedVectors.load_word2vec_format(news_path_wiki, binary=True)
    #emb_ind_glove = KeyedVectors.load_word2vec_format(glove, binary=True)
    #emb_ind_para = KeyedVectors.load_word2vec_format(paragram, binary=True)#Reading Embedding Indices

    pickle.dump(emb_ind_google, f_google)
    #pickle.dump(emb_ind_wiki, f_wiki)
    #pickle.dump(emb_ind_para, f_para)
    #pickle.dump(emb_ind_glove, f_glove)
    '''

    f_read_google = open("final_emb_data_google", "rb")
    #f_read_wiki = open("f_wiki", "rb")
    #f_read_para = open("f_para", "rb")
    #f_read_glove = open("f_glove", "rb")

    emb_r_google = pickle.load(f_read_google)
    #emb_r_wiki = pickle.load(f_read_wiki)
    #emb_r_para = pickle.load(f_read_para)
    #emb_r_glove = pickle.load(f_read_glove)

    return emb_r_google


def preprocess(df, is_train):

    #returns a list [a, b] where a => preprocessed_df ;
    #b => embeddings generated on preprocessed_df

    tqdm.pandas()
    sentences = df["question_text"].progress_apply(lambda x: x.split()).values
    vocab = build_vocab(sentences)
    print("Vocab:\n")
    print({k: vocab[k] for k in list(vocab)[:5]})

    emb_ind_google = build_embeddings()

    oov = check_coverage(vocab, emb_ind_google)
    print(oov[:20])

    df["question_text"] = df["question_text"].progress_apply(lambda x: clean_text(x))
    sentences = df["question_text"].progress_apply(lambda x: x.split())
    vocab = build_vocab(sentences)
    oov = check_coverage(vocab, emb_ind_google)
    print(oov[:20])

    df["question_text"] = df["question_text"].progress_apply(lambda x: clean_numbers(x))
    sentences = df["question_text"].progress_apply(lambda x: x.split())
    vocab = build_vocab(sentences)
    oov = check_coverage(vocab,emb_ind_google)
    print(oov[:20])

    mispell_dict = {'colour':'color',
                    'centre':'center',
                    'didnt':'did not',
                    'doesnt':'does not',
                    'isnt':'is not',
                    'shouldnt':'should not',
                    'favourite':'favorite',
                    'travelling':'traveling',
                    'counselling':'counseling',
                    'theatre':'theater',
                    'cancelled':'canceled',
                    'labour':'labor',
                    'organisation':'organization',
                    'wwii':'world war 2',
                    'citicise':'criticize',
                    'instagram': 'social medium',
                    'whatsapp': 'social medium',
                    'snapchat': 'social medium'
                   }

    mispellings, mispellings_re = _get_mispell(mispell_dict)

    df["question_text"] = df["question_text"].progress_apply(lambda x: replace_typical_misspell(x, mispellings, mispellings_re))
    sentences = df["question_text"].progress_apply(lambda x: x.split())
    to_remove = ['a','to','of','and']
    sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
    vocab = build_vocab(sentences)
    oov = check_coverage(vocab, emb_ind_google)
    print(oov[:20])

    if is_train:
        filename = "final_train_embeddings"
    else:
        filename = "final_test_embeddings"

    gen_emb_dicts(df, emb_ind_google, filename, is_train)

    return df

def gen_emb_dicts(df, emb_ind_google, dump_filename, is_train):
    outer_list = []
    c = 0
    for index, row in df.iterrows():
        c += 1
        if c % 10000 == 0:
            print("row:", c)
        d = OrderedDict()
        #l = []
        start = time.time()
        words = row["question_text"].split(" ")
        for word in words:
            try:
                d[word] = emb_ind_google[word]
            except:
                d[word] = np.zeros(300)
        #l.append(d)
        #outer_list.append(l)
        outer_list.append(d)

    if is_train:

        for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13]:
            start = time.time()
            f_name = dump_filename + "_" + str(i)
            print("Writing Data Structure to", f_name + " ...")
            file_emb = open(f_name, "wb")
            pickle.dump(outer_list[100000*i : 100000*(i + 1)], file_emb)
            file_emb.close()
            end = time.time()
            print("Finished writing to", f_name + " !")
            print("Time taken to write to file:", end - start, "seconds")
            print()

    else:
        for i in [0,1,2,3,4]:
            start = time.time()
            f_name = dump_filename + "_" + str(i)
            print("Writing Data Structure to", f_name + " ...")
            file_emb = open(f_name, "wb")
            pickle.dump(outer_list[100000*i : 100000*(i + 1)], file_emb)
            file_emb.close()
            end = time.time()
            print("Finished writing to", f_name + " !")
            print("Time taken to write to file:", end - start, "seconds")
            print()

def main():
    #Start of Codebase

    train_path = "train.csv"
    test_path = "test.csv"

    mod_train_path = "./fin_mod_train.pkl"
    mod_test_path = "./fin_mod_test.pkl"

    pickle_train = "./final_train_data.pkl"
    pickle_test = "./final_test_data.pkl"

    train_df = read_dataset(train_path)
    test_df = read_dataset(test_path)

    create_pickle(pickle_train, train_df)
    create_pickle(pickle_test, test_df)

    #train_df = load_dataset(pickle_train)
    #test_df = load_dataset(pickle_test)

    print("Preprocessing training df...")
    mod_train_df = preprocess(train_df, 1)
    print("Preprocessing testing df...")
    mod_test_df = preprocess(test_df, 0)

    create_pickle(mod_train_path, mod_train_df)
    create_pickle(mod_test_path, mod_test_df)

if __name__ == "__main__":
    main()


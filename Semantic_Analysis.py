import gensim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import os
import zipfile

def tokenize_file(file_name:str):
    tokens = []
    with open(file_name, "r", encoding="utf-8") as file:
            tokens = gensim.utils.simple_preprocess(file.read().replace("\n",""))
    return tokens

def get_dataset_tagged_documents():
    paths = ["./dataset/1of2", "./dataset/2of2"]
    files = []
    doc_num = 0
    for i in paths:
        for file_name in os.listdir(i):
            with open(i + "/" + file_name, "r", encoding="utf-8") as file:
                text = file.read().replace("\n", "") #remove newlines
                tokens = gensim.utils.simple_preprocess(text)
                files.append(gensim.models.doc2vec.TaggedDocument(tokens, [doc_num])) #Output dataset in TaggedDocument format for Doc2Vec training
                doc_num+=1
    return files

def cosine_similarity(vector1:np.ndarray, vector2:np.ndarray):
    prod_sum = 0
    vector1_power_sum = 0
    vector2_power_sum = 0
    for i in range(vector1.shape[0]):
        prod_sum += vector1[i]*vector2[i]
        vector1_power_sum += pow(vector1[i],2)
        vector2_power_sum += pow(vector2[i],2)
    return prod_sum/(math.sqrt(vector1_power_sum)*math.sqrt(vector2_power_sum))

def get_Doc2Vec_Model():
    if(not os.path.isfile("./model/WikiDoc2Vec")):
        print("Building Model...")
        documents = get_dataset_tagged_documents()
        print("- Data preprocess: done")

        model = gensim.models.Doc2Vec(vector_size=100, min_count=1, dbow_words=1)
        model.build_vocab(documents)
        print("- Vocab build: done")

        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        print("- Training: done")
        if(not os.path.exists("./model")):
            os.makedirs("./model")
        print("- Saving model to: './model/WikiDoc2Vec'")
        model.save("./model/WikiDoc2Vec")
        return model
    else:
        print("Loading model...")
        model = gensim.models.Doc2Vec.load("./model/WikiDoc2Vec")
        return model

def get_word_vectors(tokens:list, model:gensim.models.Doc2Vec):
    vectors = {}
    for word in tokens:
        vectors[word] = model.wv[word]
    return vectors

def get_dataframe_of_word_vectors(word_dict:dict):
    words = word_dict.keys()
    vector_list = []
    index_dict = {}
    i = 0
    for word in words:
        vector_list.append(word_dict[word])
        index_dict[i] = word
        i+=1
    data = np.vstack(vector_list)
    dataframe = pd.DataFrame(data)

    #scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    dataframe = pd.DataFrame(scaled_data)

    #PCA
    pca = PCA(n_components=2)
    results = pca.fit_transform(dataframe)
    principal_dataframe = pd.DataFrame(results)
    principal_dataframe.rename(index=index_dict, columns={0 : "Principal Component 1", 1 : "Principal Component 2"}, inplace=True)
    
    return principal_dataframe

def get_dataframe_of_documents(vectors:list):
    data = np.vstack(vectors)
    dataframe = pd.DataFrame(data)

    #scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    new_dataframe = pd.DataFrame(scaled_data)

    #PCA
    pca = PCA(n_components=2)
    results = pca.fit_transform(new_dataframe)

    principal_dataframe = pd.DataFrame(data=results)
    principal_dataframe.rename(columns={0: 'Principal Component 1', 1:'Principal Component 2'}, index={0 : 'Original', 1 : 'paraphraser-bart-large', 2 : 'pegasus_paraphrase', 3 : 'chatgpt_paraphraser_on_T5_base'}, inplace=True)

    return principal_dataframe

def get_figure_of_word_vectors(dataframes:list, titles:list, text_number, annotate_points = True):
    figure = plt.figure(figsize=(5,5))

    sb = figure.add_subplot(1,1,1)

    sb.set_xlabel('Principal Component 1')
    sb.set_ylabel('Principal Component 2')
    sb.set_title('PCA of Word Vectors for Text {}'.format(text_number))

    colors = ['r', 'g', 'b', 'm']

    for i in range(len(dataframes)):
        words = list(dataframes[i].index)
        x = dataframes[i].iloc[:,0]
        y = dataframes[i].iloc[:,1]
        sb.scatter(x, y, c = colors[i], s = 50, label=titles[i])
        if(annotate_points):
            for j in range(len(words)):
                sb.annotate(text=words[j], xy=(x.iloc[j],y.iloc[j]))
    sb.legend(titles)
    sb.grid()

def get_figure_of_documents(dataframe:pd.DataFrame, text_number):
    figure = plt.figure(figsize=(5,5))

    sb = figure.add_subplot(1,1,1)

    sb.set_xlabel('Principal Component 1')
    sb.set_ylabel('Principal Component 2')
    sb.set_title('PCA of Document Vectors for Text {}'.format(text_number))

    targets = list(dataframe.index)
    colors = ['r', 'g', 'b', 'm']
    for target, color in zip(targets,colors):
        sb.scatter(dataframe.loc[[target],['Principal Component 1']]
                , dataframe.loc[[target],['Principal Component 2']]
                , c = color
                , s = 50)
    sb.legend(targets)
    sb.grid()

# Start of script
versions = [["./original1.txt", "./text1_version1.txt", "./text1_version2.txt", "./text1_version3.txt"],
            ["./original2.txt", "./text2_version1.txt", "./text2_version2.txt", "./text2_version3.txt"]]
models_used = ["original", "paraphraser-bart-large", "pegasus_paraphrase", "chatgpt_paraphraser_on_T5_base"]

if((not os.path.exists("./dataset/1of2") and (not os.path.exists("./dataset/2of2")))):
    for file_name in os.listdir("./dataset"):
                with zipfile.ZipFile("./dataset/{}".format(file_name), 'r') as zip_file:
                    zip_file.extractall("./dataset")

model = get_Doc2Vec_Model()
text_num = 1
for test_text in versions:
    document_tokens = []
    for i in test_text:
        document_tokens.append(tokenize_file(i))

    document_dataframes = []
    for document in document_tokens:
        document_dataframes.append(get_dataframe_of_word_vectors(get_word_vectors(document, model)))

    get_figure_of_word_vectors(document_dataframes, models_used, text_num, False)

    doc_vectors = []
    for tokens in document_tokens:
        doc_vectors.append(model.infer_vector(tokens))

    get_figure_of_documents(get_dataframe_of_documents(doc_vectors), text_num)

    print("======Cosine similarity for Text {}======".format(text_num))
    for i in range(1,len(models_used)):
        print("Cosine similarity between ({}, {}): {}".format(models_used[0], models_used[i], cosine_similarity(doc_vectors[0],doc_vectors[i])))
    print("\n")

    text_num+=1

plt.show()
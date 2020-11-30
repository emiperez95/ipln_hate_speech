
import re
import pandas as pd
import unicodedata
import nltk
import numpy as np
import time

nltk.download('stopwords')

# ======= Data Loading and operation =======

def load_and_join_datasets(path): #ipln_hate_speech/data/
  data = pd.read_csv(path+"/train.csv", sep='\t', header=None, names=['text', 'value'], dtype={'value':bool}, engine='c')
  t_data = pd.read_csv(path+"/test.csv", sep='\t', header=None, names=['text', 'value'], dtype={'value':bool}, engine='c')
  data["origin"] = 1
  t_data["origin"] = 0
  return pd.concat([data, t_data])

def simple_data_load(path):
  return pd.read_csv(path, sep='\t', header=None, names=['text'], engine='c')

def split_datasets(data):
    ''' Split test and training data '''
    x_tr = data[data['origin'] == 1].drop(['origin'], axis=1)
    x_te = data[data['origin'] == 0].drop(['origin'], axis=1)
    y_tr = x_tr['value'].to_numpy()
    y_te = x_te['value'].to_numpy()
    x_tr.drop(['value'], axis=1, inplace=True)
    x_te.drop(['value'], axis=1, inplace=True)
    return x_tr, y_tr, x_te, y_te

def data_pipeline(text, pipe):
    for proc in pipe:
        text = proc(text)
    return text

def data_apply(data, pipe, print_res=True):
    for proc in pipe: 
        start = time.time()
        data['text'] = data['text'].apply(lambda text: proc(text))
        end = time.time()
        if print_res:
          print("\t{}: {:.4f}s".format(proc.__name__,end-start))



"""###Preprocesamiento"""

def remove_urls(text):
  return re.sub(r'(https?:\/\/\S*)|(@\S+)|(pic.twitter.com/\S+)', "", text)

def new_line_to_space(text):
  return re.sub(r'\\n', " ", text)

def remove_special_chars(text):
  return "".join([ch for ch in text if ch not in '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~¿¡@"“'])

def remove_non_ascii_chars(text):
  return "".join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))

def strip_spaces(text):
  return text.strip()

def tokenize_split(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]

# def tokenize_ntlk(text):
#     return nltk.tokenize.TweetTokenizer(strip_handles=True).tokenize(text)

def join_tokens(tokens):
    return " ".join([word for word in tokens])


"""###Corpus prunning"""

''' Remove tokens of length less than 3 '''
def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]

''' Remove spanish stopwords '''
def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('spanish')]


# """###Stemming y lematizacion"""

# def stemming_Porter(text):
#     ps = PorterStemmer()
#     return [ps.stem(word) for word in text]

# def stemming_Snowball(text):
#     ps = SnowballStemmer("spanish")
#     return [ps.stem(word) for word in text]

# def spacy_lemma(text):
#     doc = nlp(" ".join(text))
#     return [tok.lemma_ for tok in doc]


"""###Embeddings / Statistical Measures"""

def file_embedding():
    vec_size = 300
    we = pd.read_csv("data/fasttext.es.300.txt",sep=' ', header=None, engine='c')
    we.drop(we.columns[-1], axis=1, inplace=True)
    we = we.set_index(0)
    we['value'] = we[range(1,vec_size+1)].values.tolist()
    we.drop(we.columns[range(0,vec_size)], axis=1, inplace=True)
    return we.to_dict('series')['value']

"""## Interseccion embeddings con training data"""

def intersect_embedding_data(embedding, data, base_embedding=None, sobras=None):
    vec_size = 300
    if base_embedding is None:
       base_embedding = np.zeros(vec_size)

    max_len = max(data['text'].apply(lambda x: len(x)))
    emb_data = np.zeros( (len(data), max_len, vec_size) )

    for index, row in data.iterrows():
        for count, token in enumerate(row['text']):
            try:
                emb_data[index,count] = embedding[token]
            except KeyError:
                if sobras is not None:
                    sobras.append((index,token))
                emb_data[index,count] = base_embedding
    return emb_data


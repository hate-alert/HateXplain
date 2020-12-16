from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
#make sure you have downloaded the raw vectors from http://nlp.stanford.edu/data/glove.840B.300d.zip
#extract the embeddings into Data folder

# GloVe vectors loading function into temporary file
glove2word2vec('Data/glove.840B.300d.txt', 'Data/glove.840B.300d_w2v.txt')

# Load vectors directly from the file
word2vecmodel1 = KeyedVectors.load_word2vec_format('Data/glove.840B.300d_w2v.txt', binary=False)
word2vecmodel1.save("Data/word2vec.model")
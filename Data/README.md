### Encoder classes

1. classes_two.npy -  contains two classes -- "toxic" , "non-toxic"
2. classes.npy-  contains three classes -- "hatespeech" , "normal", "offensive"

Code to load the encoder
~~~
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.classes_ = np.load(<name_of_the_file>,allow_pickle=True)
~~~

### Post ids divisions
[Post_id_divisions](https://github.com/punyajoy/HateXplain/blob/master/Data/post_id_divisions.json) has a dictionary having train, valid and test post ids that are used to divide the dataset into train, val and test set in the ratio of 8:1:1.

### Word2Vec Model 
We use Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) pretrained word vectors from [glove repo](https://nlp.stanford.edu/projects/glove/). This is required only when you plan to run the non-bert deep learning model (cnn-gru, birnn, birnn-scrat). [One click download](http://nlp.stanford.edu/data/glove.840B.300d.zip)

1. Extract the glove.840B.300d.txt in this folder (Data/)
2. Run this [python file](https://github.com/punyajoy/HateXplain/blob/master/convert_to_word2vec.py) to convert the glove model into gensim model.

 :green_circle::green_circle: You are ready to roll!!! :green_circle::green_circle:





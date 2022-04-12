### Encoder classes

1. classes_two.npy -  contains two classes -- "toxic" , "non-toxic"
2. classes.npy-  contains three classes -- "hatespeech" , "normal", "offensive"

Code to load the encoder
~~~python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.classes_ = np.load(<name_of_the_file>,allow_pickle=True)
~~~

### Dataset Format ###

Sample Entry:
~~~
{
  "24198545_gab": {
    "post_id": "24198545_gab",
    "annotators": [
      {
        "label": "hatespeech",
        "annotator_id": 4,
        "target": ["African"]
      },
      {
        "label": "hatespeech",
        "annotator_id": 3,
        "target": ["African"]
      },
      {
        "label": "offensive",
        "annotator_id": 5,
        "target": ["African"]
      }
    ],
    "rationales":[
    [0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ],
    "post_tokens": ["and","this","is","why","i","end","up","with","nigger","trainee","doctors","who","can","not","speak","properly","lack","basic","knowledge","of","biology","it","truly","scary","if","the","public","only","knew"]
  }
}
~~~

:small_blue_diamond:post_id : Unique id for each post<br/>
:small_blue_diamond:annotators : The list of annotations from each annotator<br/>
:small_blue_diamond:annotators[label] : The label assigned by the annotator to this post. Possible values: [Hatespeech, Offensive, Normal]<br/>
:small_blue_diamond:annotators[annotator_id] : The unique Id assigned to each annotator<br/>
:small_blue_diamond:annotators[target] : A list of target community present in the post<br/>
:small_blue_diamond:rationales : A list of rationales selected by annotators. Each rationales represents a list with values 0 or 1. A value of 1 means that the token is part of the rationale selected by the annotator. To get the particular token, we can use the same index position in "post_tokens"<br/>
:small_blue_diamond:post_tokens : The list of tokens representing the post which was annotated<br/>


### Post ids divisions
[Post_id_divisions](https://github.com/punyajoy/HateXplain/blob/master/Data/post_id_divisions.json) has a dictionary having train, valid and test post ids that are used to divide the dataset into train, val and test set in the ratio of 8:1:1.

### Word2Vec Model 
We use Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) pretrained word vectors from [glove repo](https://nlp.stanford.edu/projects/glove/). This is required only when you plan to run the non-bert deep learning model (cnn-gru, birnn, birnn-scrat). [One click download](http://nlp.stanford.edu/data/glove.840B.300d.zip)

1. Extract the glove.840B.300d.txt in this folder (Data/)
2. Run this [python file](https://github.com/punyajoy/HateXplain/blob/master/convert_to_word2vec.py) to convert the glove model into gensim model.

 :green_circle::green_circle: You are ready to roll!!! :green_circle::green_circle:


### Note:
The data uses the label "homosexual" as defined at collection time; other sexual and gender orientation categories have been pruned from the data due to low incidence; the published version of the paper wrongly mentions the LGBTQ category.


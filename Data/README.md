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

### Dataset 

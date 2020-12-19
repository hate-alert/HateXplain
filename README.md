[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fpunyajoy%2FHateXplain&count_bg=%2379C83D&title_bg=%23555555&icon=expertsexchange.svg&icon_color=%23E7E7E7&title=Visits&edge_flat=false)](https://hits.seeyoufarm.com)
# HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection

## Abstract

Hate speech is a challenging issue plaguing the online social media. While better models for hate speech detection are continuously being developed, there is little research on the bias and interpretability aspects of hate speech. In this work, we introduce HateXplain, the first benchmark hate speech dataset covering multiple aspects of the issue. Each post in our dataset is annotated from three different perspectives: the basic, commonly used 3-class classification (i.e., hate, offensive or normal), the target community (i.e., the community that has been the victim of hate speech/offensive speech in the post), and the rationales, i.e., the portions of the post on which their labelling decision (as hate, offensive or normal) is based. We utilize existing state-of-the-art models and observe that even models that perform very well in classification do not score high on explainability metrics like model plausibility and faithfulness. We also observe that models, which utilize the human rationales for training, perform better in reducing unintended bias towards target communities. 


<p align="center"><img src="Figures/dataset_example.png" width="350" height="300"></p>

**Please cite our paper in any published work that uses any of these resources.**

~~~
@article{TO BE FILLED
}

~~~

------------------------------------------
***Folder Description*** :point_left:
------------------------------------------
~~~

./Data                --> Contains the dataset related files.
./Models              --> Contains the codes for all the classifiers used
./Preprocess  	      --> Contains the codes for preprocessing the dataset	
./best_model_json     --> Contains the parameter values for the best models

~~~
------------------------------------------
***Table of contents*** :bookmark_tabs:
------------------------------------------

:bookmark: [**Dataset**](Data/README.md) :- This describes the dataset format and setup for the dataset pipeline.
:bookmark: [**Parameters**](Parameters_description.md) :- This describes all the different parameter that are used in this code



#####  :thumbsup: The repo is still in active developements. Feel free to create an [issue](https://github.com/punyajoy/HateXplain/issues) !!  :thumbsup:

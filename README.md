[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fpunyajoy%2FHateXplain&count_bg=%2379C83D&title_bg=%23555555&icon=expertsexchange.svg&icon_color=%23E7E7E7&title=Visits&edge_flat=false)](https://hits.seeyoufarm.com)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hatexplain-a-benchmark-dataset-for/hate-speech-detection-on-hatexplain)](https://paperswithcode.com/sota/hate-speech-detection-on-hatexplain?p=hatexplain-a-benchmark-dataset-for)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![Generic badge](https://img.shields.io/badge/Dataset-Hatexplain-red.svg)](https://huggingface.co/datasets/hatexplain)
[![Generic badge](https://img.shields.io/badge/Models-Hatexplain-blue.svg)](https://huggingface.co/models?search=hatexplain)


# :mag_right: HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection [Accepted at AAAI 2021]

### :tada: :tada: BERT for detecting abusive language(Hate speech+offensive) and predicting rationales is uploaded [here](https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two). Be sure to check it out :tada: :tada:.

#### For more details about our paper

Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, and Animesh Mukherjee "[HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection](https://ojs.aaai.org/index.php/AAAI/article/view/17745)". Accepted at AAAI 2021.

[Arxiv paper link](https://arxiv.org/abs/2012.10289)
# Abstract

Hate speech is a challenging issue plaguing the online social media. While better models for hate speech detection are continuously being developed, there is little research on the bias and interpretability aspects of hate speech. In this work, we introduce HateXplain, the first benchmark hate speech dataset covering multiple aspects of the issue. Each post in our dataset is annotated from three different perspectives: the basic, commonly used 3-class classification (i.e., hate, offensive or normal), the target community (i.e., the community that has been the victim of hate speech/offensive speech in the post), and the rationales, i.e., the portions of the post on which their labelling decision (as hate, offensive or normal) is based. We utilize existing state-of-the-art models and observe that even models that perform very well in classification do not score high on explainability metrics like model plausibility and faithfulness. We also observe that models, which utilize the human rationales for training, perform better in reducing unintended bias towards target communities. 

***WARNING: The repository contains content that are offensive and/or hateful in nature.***

<p align="center"><img src="Figures/dataset_example.png" width="350" height="300"></p>

**Please cite our paper in any published work that uses any of these resources.**

~~~bibtex

@inproceedings{mathew2021hatexplain,
  title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
  author={Mathew, Binny and Saha, Punyajoy and Yimam, Seid Muhie and Biemann, Chris and Goyal, Pawan and Mukherjee, Animesh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={17},
  pages={14867--14875},
  year={2021}
}

~~~

------------------------------------------
***Folder Description*** :open_file_folder:	
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

------------------------------------------
***Usage instructions*** 
------------------------------------------
please setup the **Dataset** first (more important if your using non-bert model). Install the libraries using the following command (preferably inside an environemt)
~~~
pip install -r requirements.txt
~~~
#### Training
To train the model use the following command.
~~~
usage: manual_training_inference.py [-h]
                                    --path_to_json --use_from_file
                                    --attention_lambda

Train a deep-learning model with the given data

positional arguments:
  --path_to_json      The path to json containining the parameters
  --use_from_file     whether use the parameters present here or directly use
                      from file
  --attention_lambda  required to assign the contribution of the atention loss

~~~
You can either set the parameters present in the python file, option will be (--use_from_file set to True). To change the parameters, check the **Parameters** section for more details. The code will run on CPU by default. The recommended way will be to copy one of the dictionary in `best_model_json` and change it accordingly.

* **For transformer models** :-The repository current supports the model having similar tokenization as [BERT](https://huggingface.co/transformers/model_doc/bert.html). In the params set `bert_tokens` to True and `path_files` to any of BERT based models in [Huggingface](https://huggingface.co/). 
* **For non-transformer models** :-The repository current supports the LSTM, LSTM attention and CNN GRU models. In the params set `bert_tokens` to False and model name according to **Parameters** section (either birnn, birnnatt, birnnscrat, cnn_gru).

For more details about the end to end pipleline visit [our_demo](https://github.com/punyajoy/HateXplain/blob/master/Example_HateExplain.ipynb)

### Blogs and github repos which we used for reference :angel:
1. For finetuning BERT this [blog](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)  by Chris McCormick is used and we also referred [Transformers github repo](https://github.com/huggingface/transformers).
2. For CNN-GRU model we used the original [repo](https://github.com/ziqizhang/chase) for reference.
3. For Evaluation using the Explanantion metrics we used the ERASER benchmark [repo](https://github.com/jayded/eraserbenchmark). Please look into their repo and paper for more details.


### Todos
- [x] Add arxiv paper link and description.
- [ ] Release better documentation for Models and Preprocess sections.
- [ ] Add other Transformers model to the pipeline.
- [x] Upload our model to [transformers community](https://huggingface.co/models) to make them public
- [x] Create an interface for **social scientists** where they can use our models easily with their data


#####  :thumbsup: The repo is still in active developements. Feel free to create an [issue](https://github.com/punyajoy/HateXplain/issues) !!  :thumbsup:

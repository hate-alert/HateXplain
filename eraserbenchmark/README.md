# eraserbenchmark
A benchmark for understanding and evaluating rationales: http://www.eraserbenchmark.com/

## Core Files

The core files are [utils](rationale_benchmark/utils.py) and [metrics](rationale_benchmark/metrics.py).
These two files comprise everything you need to work with our released datasets.

[utils](rationale_benchmark/utils.py) documents everything you need to know about our input formats. Output
formats and validation code are covered in [metrics](rationale_benchmark/metrics.py).

## Models

At the moment we offer two forms of pipeline models:
* (Lehman, et al., 2019) - sentence level rationale identification, followed by taking the best resulting sentence and classifying it.
    * Both portions of this pipeline function via encoding the input sentence (via a GRU), attending (conditioned on a query vector), and making a classification.
* BERT-To-BERT - the same as above, but using a BERT model.

### (Lehman, et al., 2019) Pipeline

To run this model, we need to first:
* create a `model_components`, `data`, and `output`, directory
* download GloVe vectors from http://nlp.stanford.edu/data/glove.6B.zip and extract the 200 dimensional vector to `model_components`
* download http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin to `model_components`
* set up a virtual env meeting requirements.txt
* download data from the primary website to `data` and extract each dataset to its respective directory
* ensure that we have at least an 11G GPU. Reducing batch sizes may enable running on a smaller GPU.

Then we can run (as an example):
```
PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/models/pipeline/pipeline_train.py --data_dir data/movies --output_dir output/movies --model_params params/movies.json
PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/metrics.py --split test --data_dir data/movies --results output/movies/test_decoded.jsonl --score_file output/movies/test_scores.json
```

### BERT-To-BERT Pipeline

To run this model, instructions are effectively the same as the simple pipeline above, except we also require a GPU with approximately 16G of memory (e.g. Tesla V100). The same caveats about batch sizes apply here as well.

Then we can run (as an example):
```
PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/models/pipeline/bert_pipeline.py --data_dir data/movies --output_dir output_bert/movies --model_params param/movies.json
PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/metrics.py --split test --data_dir data/movies --results output_bert/movies/test_decoded.jsonl --score_file output_bert/movies/test_scores.json
```

For more examples, see the [BERT-to-BERT reproduction](REPRODUCTION.txt).

More models including Lei et al can be found at : https://github.com/successar/Eraser-Benchmark-Baseline-Models

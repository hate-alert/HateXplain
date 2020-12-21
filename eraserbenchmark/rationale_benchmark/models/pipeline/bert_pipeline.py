# TODO consider if this can be collapsed back down into the pipeline_train.py
import argparse
import json
import logging
import random
import os

from itertools import chain
from typing import List, Tuple

import numpy as np
import torch

from transformers import BertTokenizer

from rationale_benchmark.models.sequence_taggers import BertTagger
from rationale_benchmark.utils import (
    Annotation,
    Evidence,
    write_jsonl,
    load_datasets,
    load_documents,
)
from rationale_benchmark.models.mlp import (
    BertClassifier,
)
from rationale_benchmark.models.pipeline.evidence_identifier import train_evidence_identifier
from rationale_benchmark.models.pipeline.evidence_token_identifier import train_evidence_token_identifier
from rationale_benchmark.models.pipeline.evidence_classifier import train_evidence_classifier
from rationale_benchmark.models.pipeline.pipeline_utils import decode, decode_evidence_tokens_and_classify

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)
# let's make this more or less deterministic (not resistent to restarts)
random.seed(12345)
np.random.seed(67890)
torch.manual_seed(10111213)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def initialize_models(params: dict, batch_first: bool, unk_token='<unk>'):
    assert batch_first
    max_length = params['max_length']
    tokenizer = BertTokenizer.from_pretrained(params['bert_vocab'])
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    bert_dir = params['bert_dir']
    if bool(params['use_evidence_sentence_identifier']):
        use_half_precision = bool(params['evidence_identifier'].get('use_half_precision', 1))
        evidence_identifier = BertClassifier(bert_dir=bert_dir,
                                             pad_token_id=pad_token_id,
                                             cls_token_id=cls_token_id,
                                             sep_token_id=sep_token_id,
                                             num_labels=2,
                                             max_length=max_length,
                                             use_half_precision=use_half_precision)
    else:
        evidence_identifier = None

    if bool(params['use_evidence_token_identifier']):
        use_half_precision = bool(params['evidence_token_identifier'].get('use_half_precision', 1))
        evidence_token_identifier = BertTagger(bert_dir=bert_dir,
                                               pad_token_id=pad_token_id,
                                               cls_token_id=cls_token_id,
                                               sep_token_id=sep_token_id,
                                               max_length=max_length,
                                               use_half_precision=use_half_precision)
    else:
        evidence_token_identifier = None
    use_half_precision = bool(params['evidence_classifier'].get('use_half_precision', 1))
    evidence_classes = dict((y, x) for (x, y) in enumerate(params['evidence_classifier']['classes']))
    evidence_classifier = BertClassifier(bert_dir=bert_dir,
                                         pad_token_id=pad_token_id,
                                         cls_token_id=cls_token_id,
                                         sep_token_id=sep_token_id,
                                         num_labels=len(evidence_classes),
                                         max_length=max_length,
                                         use_half_precision=use_half_precision)
    word_interner = tokenizer.vocab
    de_interner = tokenizer.ids_to_tokens
    return evidence_identifier, evidence_token_identifier, evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer


def bert_tokenize_doc(doc: List[List[str]], tokenizer, special_token_map) -> Tuple[List[List[str]], List[List[Tuple[int, int]]]]:
    """ Tokenizes a document and returns [start, end) spans to map the wordpieces back to their source words"""
    sents = []
    sent_token_spans = []
    for sent in doc:
        tokens = []
        spans = []
        start = 0
        for w in sent:
            if w in special_token_map:
                tokens.append(w)
            else:
                tokens.extend(tokenizer.tokenize(w))
            end = len(tokens)
            spans.append((start, end))
            start = end
        sents.append(tokens)
        sent_token_spans.append(spans)
    return sents, sent_token_spans


def bert_intern_doc(doc: List[List[str]], tokenizer, special_token_map) -> List[List[int]]:
    return [list(chain.from_iterable(special_token_map.get(w, tokenizer.encode(w)) for w in s)) for s in doc]


def bert_intern_annotation(annotations: List[Annotation], tokenizer):
    ret = []
    for ann in annotations:
        ev_groups = []
        for ev_group in ann.evidences:
            evs = []
            for ev in ev_group:
                text = list(chain.from_iterable(tokenizer.tokenize(w) for w in ev.text.split()))
                if len(text) == 0:
                    continue
                text = tokenizer.encode(text, add_special_tokens=False)
                evs.append(Evidence(text=tuple(text),
                                    docid=ev.docid,
                                    start_token=ev.start_token,
                                    end_token=ev.end_token,
                                    start_sentence=ev.start_sentence,
                                    end_sentence=ev.end_sentence))
            ev_groups.append(tuple(evs))
        query = list(chain.from_iterable(tokenizer.tokenize(w) for w in ann.query.split()))
        if len(query) > 0:
            query = tokenizer.encode(query, add_special_tokens=False)
        else:
            query = []
        ret.append(Annotation(annotation_id=ann.annotation_id,
                              query=tuple(query),
                              evidences=frozenset(ev_groups),
                              classification=ann.classification,
                              query_type=ann.query_type))
    return ret


BATCH_FIRST = True


def main():
    parser = argparse.ArgumentParser(description="""Trains a pipeline model.

    Step 1 is evidence identification, that is identify if a given sentence is evidence or not
    Step 2 is evidence classification, that is given an evidence sentence, classify the final outcome for the final task
     (e.g. sentiment or significance).

    These models should be separated into two separate steps, but at the moment:
    * prep data (load, intern documents, load json)
    * convert data for evidence identification - in the case of training data we take all the positives and sample some
      negatives
        * side note: this sampling is *somewhat* configurable and is done on a per-batch/epoch basis in order to gain a
          broader sampling of negative values.
    * train evidence identification
    * convert data for evidence classification - take all rationales + decisions and use this as input
    * train evidence classification
    * decode first the evidence, then run classification for each split
    
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--model_params', dest='model_params', required=True,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    args = parser.parse_args()
    assert BATCH_FIRST
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.model_params, 'r') as fp:
        logger.info(f'Loading model parameters from {args.model_params}')
        model_params = json.load(fp)
        logger.info(f'Params: {json.dumps(model_params, indent=2, sort_keys=True)}')
    train, val, test = load_datasets(args.data_dir)
    docids = set(e.docid for e in
                 chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
    documents = load_documents(args.data_dir, docids)
    logger.info(f'Loaded {len(documents)} documents')
    # this ignores the case where annotations don't align perfectly with token boundaries, but this isn't that important
    unk_token = '<unk>'
    evidence_identifier, evidence_token_identifier, evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer = \
        initialize_models(model_params,
                          batch_first=BATCH_FIRST,
                          unk_token=unk_token)
    if not ((evidence_identifier is None) ^ (evidence_token_identifier is None)):
        raise ValueError('Exactly one of the evidence identifier and evidence token identifier must be defined, not both!')
    logger.info(f'We have {len(word_interner)} wordpieces')
    cache = os.path.join(args.output_dir, 'preprocessed.pkl')
    if os.path.exists(cache):
        logger.info(f'Loading interned documents from {cache}')
        (interned_documents, interned_document_token_slices) = torch.load(cache)
    else:
        logger.info(f'Interning documents')
        special_token_map =  {
                                 'SEP': [evidence_classifier.sep_token_id],
                                 '[SEP]': [evidence_classifier.sep_token_id],
                                 '[sep]': [evidence_classifier.sep_token_id],
                                 'UNK': [tokenizer.unk_token_id],
                                 '[UNK]': [tokenizer.unk_token_id],
                                 '[unk]': [tokenizer.unk_token_id],
                                 'PAD': [tokenizer.unk_token_id],
                                 '[PAD]': [tokenizer.unk_token_id],
                                 '[pad]': [tokenizer.unk_token_id],
                             }
        interned_documents = {}
        interned_document_token_slices = {}
        for d, doc in documents.items():
            tokenized, w_slices = bert_tokenize_doc(doc, tokenizer, special_token_map=special_token_map)
            interned_documents[d] = bert_intern_doc(tokenized, tokenizer, special_token_map=special_token_map)
            interned_document_token_slices[d] = w_slices
        torch.save((interned_documents, interned_document_token_slices), cache)
    interned_train = bert_intern_annotation(train, tokenizer)
    interned_val = bert_intern_annotation(val, tokenizer)
    interned_test = bert_intern_annotation(test, tokenizer)

    # train the evidence identifier
    if evidence_identifier is not None:
        logger.info('Beginning training of the evidence sentence identifier')
        evidence_identifier = evidence_identifier.cuda()
        optimizer = None
        scheduler = None
        evidence_identifier, evidence_ident_results = train_evidence_identifier(evidence_identifier,
                                                                                args.output_dir,
                                                                                interned_train,
                                                                                interned_val,
                                                                                interned_documents,
                                                                                model_params,
                                                                                optimizer=optimizer,
                                                                                scheduler=scheduler,
                                                                                tensorize_model_inputs=True)
        evidence_identifier = evidence_identifier.cpu()  # free GPU space for next model
    else:
        logger.info('No evidence sentence identifier provided; skipping')
        evidence_identifier, evidence_ident_results = None, None

    if evidence_token_identifier is not None:
        logger.info('Beginning training of the evidence token identifier')
        evidence_token_identifier = evidence_token_identifier.cuda()
        evidence_token_identifier, evidence_token_identifier_results = \
            train_evidence_token_identifier(evidence_token_identifier,
                                            args.output_dir,
                                            interned_train,
                                            interned_val,
                                            interned_documents=interned_documents,
                                            source_documents=documents,
                                            token_mapping=interned_document_token_slices,
                                            model_pars=model_params,
                                            tensorize_model_inputs=True
                                            )
        evidence_token_identifier = evidence_token_identifier.cpu()
    else:
        logger.info('No evidence token identifier provided; skipping')
        evidence_token_identifier_results = None

    # train the evidence classifier
    logger.info('Beginning training of the evidence classifier')
    evidence_classifier = evidence_classifier.cuda()
    optimizer = None
    scheduler = None
    evidence_classifier, evidence_class_results = train_evidence_classifier(evidence_classifier,
                                                                            args.output_dir,
                                                                            interned_train,
                                                                            interned_val,
                                                                            interned_documents,
                                                                            model_params,
                                                                            optimizer=optimizer,
                                                                            scheduler=scheduler,
                                                                            class_interner=evidence_classes,
                                                                            tensorize_model_inputs=True,
                                                                            token_only_evidence=(evidence_token_identifier is not None))

    # decode
    logger.info('Beginning final decoding')
   
    if evidence_identifier is not None:
        pipeline_batch_size = min([model_params['evidence_classifier']['batch_size'],
                                   model_params['evidence_identifier']['batch_size']])
        pipeline_results, train_decoded, val_decoded, test_decoded = decode(evidence_identifier.cuda(),
                                                                            evidence_classifier.cuda(),
                                                                            interned_train,
                                                                            interned_val,
                                                                            interned_test,
                                                                            interned_documents,
                                                                            evidence_classes,
                                                                            pipeline_batch_size,
                                                                            BATCH_FIRST,
                                                                            decoding_docs=documents)
    elif evidence_token_identifier is not None:
        pipeline_batch_size = min([model_params['evidence_classifier']['batch_size'],
                                   model_params['evidence_token_identifier']['batch_size']])
        use_cose_hack = bool(model_params['evidence_token_identifier'].get('cose_data_hack', 0))
        if use_cose_hack:
            logger.info("Using COS-E data prep and processing hacks!")
            logger.info("These hacks impact evidence identification, classification, and decoding. They are not general")
        pipeline_results, train_decoded, val_decoded, test_decoded = decode_evidence_tokens_and_classify(evidence_token_identifier.cuda(),
                                                                                                         evidence_classifier.cuda(),
                                                                                                         interned_train,
                                                                                                         interned_val,
                                                                                                         interned_test,
                                                                                                         interned_documents,
                                                                                                         documents,
                                                                                                         interned_document_token_slices,  # token_mapping
                                                                                                         evidence_classes,
                                                                                                         pipeline_batch_size,
                                                                                                         decoding_docs=documents,
                                                                                                         use_cose_hack=use_cose_hack)
    else:
        raise StateError('Impossible state!')
        
    write_jsonl(train_decoded, os.path.join(args.output_dir, 'train_decoded.jsonl'))
    write_jsonl(val_decoded, os.path.join(args.output_dir, 'val_decoded.jsonl'))
    write_jsonl(test_decoded, os.path.join(args.output_dir, 'test_decoded.jsonl'))
    with open(os.path.join(args.output_dir, 'identifier_results.json'), 'w') as ident_output, \
            open(os.path.join(args.output_dir, 'classifier_results.json'), 'w') as class_output:
        ident_output.write(json.dumps(evidence_ident_results))
        class_output.write(json.dumps(evidence_class_results))
    for k, v in pipeline_results.items():
        if type(v) is dict:
            for k1, v1 in v.items():
                logging.info(f'Pipeline results for {k}, {k1}={v1}')
        else:
            logging.info(f'Pipeline results {k}\t={v}')


if __name__ == '__main__':
    main()

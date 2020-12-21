import argparse
import json
import logging
import random
import os

from collections import defaultdict, OrderedDict
from heapq import nsmallest, nlargest
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

#import apex
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, classification_report

from rationale_benchmark.utils import (
    Annotation,
    load_datasets,
    load_documents,
    intern_documents,
    intern_annotations
)
from rationale_benchmark.models.model_utils import (
    PaddedSequence,
    extract_embeddings,
)
from rationale_benchmark.models.mlp import (
    AttentiveClassifier,
    LuongAttention,
    RNNEncoder,
    WordEmbedder
)

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
# let's make this more or less deterministic (not resistent to restarts)
random.seed(12345)
np.random.seed(67890)
torch.manual_seed(10111213)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def initialize_model(params: dict, vocab: Set[str], batch_first: bool, unk_token='UNK'):
    # TODO this is obviously asking for some sort of dependency injection. implement if it saves me time.
    if 'embedding_file' in params['embeddings']:
        embeddings, word_interner, de_interner = extract_embeddings(vocab, params['embeddings']['embedding_file'],
                                                                    unk_token=unk_token)
        if torch.cuda.is_available():
            embeddings = embeddings.cuda()
    else:
        raise ValueError("No 'embedding_file' found in params!")
    word_embedder = WordEmbedder(embeddings, params['embeddings']['dropout'])
    encoding_size = params['classifier'].get('hidden_size', word_embedder.output_dimension)
    if bool(params['classifier']['has_query']):
        attention_mechanism = LuongAttention(encoding_size)
        query_encoder = RNNEncoder(word_embedder,
                                   batch_first=batch_first,
                                   condition=False,
                                   output_dimension=encoding_size,
                                   attention_mechanism=attention_mechanism)
        condition = True
        query_size = query_encoder.output_dimension
    else:
        query_encoder = None
        condition = False
        query_size = None
    attention_mechanism = LuongAttention(encoding_size, encoding_size)
    document_encoder = RNNEncoder(word_embedder,
                                  batch_first=batch_first,
                                  condition=condition,
                                  output_dimension=encoding_size,
                                  attention_mechanism=attention_mechanism)
    evidence_classes = dict((y, x) for (x, y) in enumerate(params['classifier']['classes']))
    classifier = AttentiveClassifier(document_encoder,
                                     query_encoder,
                                     len(evidence_classes),
                                     params['classifier']['mlp_size'],
                                     params['classifier']['dropout'])
    return classifier, word_interner, de_interner, evidence_classes


def annotation_to_instances(ann: Annotation, docs: Dict[str, List[List[int]]], class_interner: Dict[str, int]):
    evidences = defaultdict(set)
    for ev in ann.all_evidences():
        evidences[ev.docid].add(ev)
    output_documents = dict()
    evidence_spans = dict()
    for d, evs in evidences.items():
        output_documents[d] = list(chain.from_iterable(docs[d]))
        evidence_targets = [0 for _ in range(sum(len(s) for s in docs[d]))]
        for ev in evs:
            for t in range(ev.start_token, ev.end_token):
                evidence_targets[t] = 1
        evidence_spans[d] = evidence_targets
    return class_interner.get(ann.classification, -1), output_documents, evidence_spans


def convert_for_training(annotations: List[Annotation], docs: Dict[str, List[List[int]]],
                         class_interner: Dict[str, int]):
    ids = []
    classes = []
    queries = []
    doc_vecs = []
    evidence_spans = []
    for ann in annotations:
        kls, flattened_docs, ev_spans = annotation_to_instances(ann, docs, class_interner)
        if len(flattened_docs) == 0:
            continue
        if ann.query and len(ann.query) > 0:
            queries.append(torch.tensor(ann.query))
        classes.append(kls)
        ids.append((ann.annotation_id, flattened_docs.keys()))
        combined_doc_vecs = []
        combined_evidence_spans = []
        for d, doc_vec in flattened_docs.items():
            combined_doc_vecs.extend(doc_vec)
            combined_evidence_spans.extend(ev_spans[d])
        doc_vecs.append(torch.tensor(combined_doc_vecs))
        evidence_spans.append(torch.tensor(combined_evidence_spans))
    if len(queries) == 0:
        queries = None
    return ids, classes, queries, doc_vecs, evidence_spans


def train_classifier(classifier: nn.Module,
                     save_dir: str,
                     train: List[Annotation],
                     val: List[Annotation],
                     documents: Dict[str, List[List[int]]],
                     model_pars: dict,
                     class_interner: Dict[str, int],
                     attention_optimizer=None,
                     classifier_optimizer=None) -> Tuple[nn.Module, dict]:
    logging.info(f'Beginning training classifier with {len(train)} annotations, {len(val)} for validation')
    # TODO this paramterization is a mess and all parameters should be easier to track
    classifier_output_dir = os.path.join(save_dir, 'classifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(classifier_output_dir, exist_ok=True)
    model_save_file = os.path.join(classifier_output_dir, 'classifier.pt')
    epoch_save_file = os.path.join(classifier_output_dir, 'classifier_epoch_data.pt')
    train_ids, train_classes, train_queries, train_docs, train_evidence_spans = convert_for_training(train,
                                                                                                     documents,
                                                                                                     class_interner)
    val_ids, val_classes, val_queries, val_docs, val_evidence_spans = convert_for_training(val,
                                                                                           documents,
                                                                                           class_interner)
    if not bool(model_pars['classifier']['has_query']):
        train_queries = None
        val_queries = None
    device = next(classifier.parameters()).device

    if attention_optimizer is None:
        attention_optimizer = torch.optim.Adam(classifier.parameters(), lr=model_pars['classifier']['lr'])
    if classifier_optimizer is None:
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=model_pars['classifier']['lr'])
    attention_criterion = nn.BCELoss(reduction='sum')
    criterion = nn.CrossEntropyLoss(reduction='sum')
    batch_size = model_pars['classifier']['batch_size']
    epochs = model_pars['classifier']['epochs']
    attention_epochs = model_pars['classifier']['attention_epochs']
    patience = model_pars['classifier']['patience']
    max_grad_norm = model_pars['classifier'].get('max_grad_norm', None)
    class_labels = [k for k, v in sorted(class_interner.items())]

    results = {
        'attention_train_losses': [],
        'attention_val_losses': [],
        'train_loss': [],
        'train_f1': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': [],
    }
    best_attention_epoch = -1
    best_classifier_epoch = -1
    best_attention_loss = float('inf')
    best_classifier_loss = float('inf')
    best_model_state_dict = None
    start_attention_epoch = 0
    start_classifier_epoch = 0
    epoch_data = {}
    if os.path.exists(epoch_save_file):
        logging.info(f'Restoring model from {model_save_file}')
        classifier.load_state_dict(torch.load(model_save_file))
        epoch_data = torch.load(epoch_save_file)
        start_attention_epoch = epoch_data.get('attention_epoch', -1) + 1
        start_classifier_epoch = epoch_data.get('classifier_epoch', -1) + 1
        best_attention_loss = epoch_data.get('best_attention_loss', float('inf'))
        best_classifier_loss = epoch_data.get('best_classifier_loss', float('inf'))
        # handle finishing because patience was exceeded or we didn't get the best final epoch
        if bool(epoch_data.get('done_attention', 0)):
            start_attention_epoch = epochs
        if bool(epoch_data.get('done_classifier', 0)):
            start_classifier_epoch = epochs
        results = epoch_data['results']
        best_attention_epoch = start_attention_epoch
        best_classifier_epoch = start_classifier_epoch
        best_model_state_dict = OrderedDict({k: v.cpu() for k, v in classifier.state_dict().items()})
        logging.info(f'Restoring training from attention epoch {start_attention_epoch} / {start_classifier_epoch}')
    logging.info(f'Training classifier attention from epoch {start_attention_epoch} until epoch {attention_epochs}')
    for attention_epoch in range(start_attention_epoch, attention_epochs):
        epoch_train_loss = 0
        epoch_train_tokens = 0
        epoch_val_loss = 0
        epoch_val_tokens = 0
        for batch_start in range(0, len(train_ids), batch_size):
            #targets = train_classes[batch_start:batch_start + batch_size]
            classifier.train()
            attention_optimizer.zero_grad()
            if train_queries is None:
                queries = None
            else:
                queries = train_queries[batch_start:batch_start + batch_size]
            docs = train_docs[batch_start:batch_start + batch_size]
            train_spans = train_evidence_spans[batch_start:batch_start + batch_size]
            _, _, _, unnormalized_document_attention, _ = classifier(queries, None, docs, return_attentions=True)
            partially_normalized_document_attention = torch.sigmoid(unnormalized_document_attention.data.squeeze())
            train_spans = PaddedSequence.autopad(train_spans, batch_first=True,
                                                 device=unnormalized_document_attention.data.device)
            batch_loss = attention_criterion(partially_normalized_document_attention, train_spans.data.float())
            epoch_train_loss += batch_loss.item()
            train_size = torch.sum(train_spans.batch_sizes).item()
            epoch_train_tokens += train_size
            batch_loss = batch_loss / train_size
            batch_loss.backward()
            attention_optimizer.step()
        results['attention_train_losses'].append(epoch_train_loss / epoch_train_tokens)
        logging.info(f'Epoch {attention_epoch} attention train loss {epoch_train_loss / epoch_train_tokens}')
        with torch.no_grad():
            classifier.eval()
            for batch_start in range(0, len(val_ids), batch_size):
                #targets = val_classes[batch_start:batch_start + batch_size]
                if val_queries is None:
                    queries = None
                else:
                    queries = val_queries[batch_start:batch_start + batch_size]
                docs = val_docs[batch_start:batch_start + batch_size]
                val_spans = val_evidence_spans[batch_start:batch_start + batch_size]
                _, _, _, unnormalized_document_attention, _ = classifier(queries, None, docs, return_attentions=True)
                unnormalized_document_attention = torch.sigmoid(unnormalized_document_attention.data)
                val_spans = PaddedSequence.autopad(val_spans, batch_first=True, device=device)
                batch_loss = attention_criterion(unnormalized_document_attention.squeeze(), val_spans.data.float())
                epoch_val_loss += batch_loss.item()
                epoch_val_tokens += torch.sum(val_spans.batch_sizes).item()
            epoch_val_loss = epoch_val_loss / epoch_val_tokens
            results['attention_val_losses'].append(epoch_val_loss)
            logging.info(f'Epoch {attention_epoch} attention val loss {epoch_val_loss}')
            if epoch_val_loss < best_attention_loss:
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in classifier.state_dict().items()})
                best_attention_epoch = attention_epoch
                best_attention_loss = epoch_val_loss
                epoch_data['attention_epoch'] = attention_epoch
                epoch_data['results'] = results
                epoch_data['best_attention_loss'] = best_attention_loss
                epoch_data['best_classifier_loss'] = float('inf')
                epoch_data['done_attention'] = 0
                epoch_data['done_classifier'] = 0
                torch.save(classifier.state_dict(), model_save_file)
                torch.save(epoch_data, epoch_save_file)
                logging.info(f'Epoch {attention_epoch} new best model with val loss {epoch_val_loss}')
        if attention_epoch - best_attention_epoch > patience:
            logging.info(f'Exiting after epoch {attention_epoch} due to no improvement')
            epoch_data['done_attention'] = 1
            torch.save(epoch_data, epoch_save_file)
            break
    logging.info(f'Training classifier from epoch {start_classifier_epoch} until epoch {epochs}')
    for classifier_epoch in range(start_classifier_epoch, epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0
        train_preds = []
        train_truth = []
        classifier.train()
        for batch_start in range(0, len(train_ids), batch_size):
            classifier.train()
            classifier_optimizer.zero_grad()
            targets = train_classes[batch_start:batch_start + batch_size]
            train_truth.extend(targets)
            targets = torch.tensor(targets, device=device)
            if train_queries is not None:
                queries = train_queries[batch_start:batch_start + batch_size]
            else:
                queries = None
            docs = train_docs[batch_start:batch_start + batch_size]
            classes = classifier(queries, None, docs, return_attentions=False)
            train_preds.extend(x.item() for x in torch.argmax(classes, dim=1))
            batch_loss = criterion(classes.squeeze(), targets)
            epoch_train_loss += batch_loss.item()
            batch_loss /= len(docs)
            batch_loss.backward()
            classifier_optimizer.step()
        train_accuracy = accuracy_score(train_truth, train_preds)
        train_f1 = classification_report(train_truth, train_preds, output_dict=True)
        results['train_loss'].append(epoch_train_loss / len(train_ids))
        results['train_acc'].append(train_accuracy)
        results['train_f1'].append(train_f1)
        logging.info(
            f'Epoch {classifier_epoch} train loss {epoch_train_loss / len(train_ids)}, accuracy: {train_accuracy}, f1: {train_f1}')
        with torch.no_grad():
            classifier.eval()
            val_preds = []
            val_truth = []
            for batch_start in range(0, len(val_ids), batch_size):
                targets = val_classes[batch_start:batch_start + batch_size]
                val_truth.extend(targets)
                if val_queries is not None:
                    queries = val_queries[batch_start:batch_start + batch_size]
                else:
                    queries = None
                docs = val_docs[batch_start:batch_start + batch_size]
                classes = classifier(queries, None, docs, return_attentions=False)
                targets = torch.tensor(targets, device=classes.device)
                val_preds.extend(x.item() for x in torch.argmax(classes, dim=1))
                batch_loss = criterion(classes, targets)
                if not torch.all(batch_loss == batch_loss):
                    import pdb; pdb.set_trace()
                epoch_val_loss += batch_loss.item()
                batch_loss /= len(docs)
            epoch_val_loss /= len(val_ids)
            val_accuracy = accuracy_score(val_truth, val_preds)
            val_f1 = classification_report(val_truth, val_preds, output_dict=True)
            results['val_loss'].append(epoch_val_loss)
            results['val_acc'].append(val_accuracy)
            results['val_f1'].append(val_f1)
            logging.info(f'Epoch {classifier_epoch} val loss {epoch_val_loss}, accuracy: {val_accuracy}, f1: {val_f1}')
            if epoch_val_loss < best_classifier_loss:
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in classifier.state_dict().items()})
                best_classifier_epoch = classifier_epoch
                best_val_loss = epoch_val_loss
                epoch_data['classifier_epoch'] = classifier_epoch
                epoch_data['attention_epoch'] = best_attention_epoch
                epoch_data['best_attention_loss'] = best_attention_loss
                epoch_data['results'] = results
                epoch_data['best_classifier_loss'] = best_val_loss
                epoch_data['done_classifier'] = 0
                epoch_data['done_attention'] = 1
                torch.save(classifier.state_dict(), model_save_file)
                torch.save(epoch_data, epoch_save_file)
                logging.info(f'Epoch {classifier_epoch} new best model with val loss {epoch_val_loss}')
        if classifier_epoch - best_classifier_epoch > patience:
            logging.info(f'Exiting after epoch {classifier_epoch} due to no improvement')
            epoch_data['done_classifier'] = 1
            torch.save(epoch_data, epoch_save_file)
            break
    return classifier, results


def decode(classifier: nn.Module,
           train: List[Annotation],
           val: List[Annotation],
           test: List[Annotation],
           documents: Dict[str, List[List[int]]],
           class_interner: Dict[str, int],
           batch_size: int,
           tensorize_model_inputs: bool,
           threshold: float,
           k_fraction: float,
           has_query: bool) -> dict:
    class_deinterner = {v: k for k, v in class_interner.items()}
    train_ids, train_classes, train_queries, train_docs, train_evidence_spans = convert_for_training(train, documents,
                                                                                                     class_interner)
    val_ids, val_classes, val_queries, val_docs, val_evidence_spans = convert_for_training(val, documents,
                                                                                           class_interner)
    test_ids, _, test_queries, test_docs, test_evidence_spans = convert_for_training(test, documents, class_interner)
    if not bool(has_query):
        train_queries = None
        val_queries = None
        test_queries = None
    device = next(classifier.parameters()).device

    def decode_set(queries, docs):
        classifier.eval()
        with torch.no_grad():
            preds, unnormalized_attentions, attentions = [], [], []
            for batch_start in range(0, len(docs), batch_size):
                if queries is not None:
                    q = queries[batch_start:batch_start + batch_size]
                    q = [torch.tensor(x) for x in q]
                else:
                    q = None
                d = docs[batch_start:batch_start + batch_size]
                d = [torch.tensor(x) for x in d]
                classes, _, _, unnormalized_document_attention, normalized_document_attention = classifier(q, None, d,
                                                                                                           return_attentions=True)
                preds.extend([[y.item() for y in x] for x in classes])
                attentions.extend([[x.item() for x in y] for y in
                                   normalized_document_attention.unpad(normalized_document_attention.data.squeeze())])
                unnormalized_attentions.extend([[x.item() for x in y] for y in unnormalized_document_attention.unpad(
                    unnormalized_document_attention.data.squeeze())])
                #unnormalized_attentions.append([x.item() for x in unnormalized_document_attention])
        return preds, attentions

    def generate_rationales(ids, queries, docs):
        rats = []
        evidence_only_docs = []
        no_evidence_docs = []
        for (ann_id, docids), doc, pred, attentions in zip(ids, docs, *decode_set(queries, docs)):
            doc = np.array(doc)
            classification_scores = {class_deinterner[cls]: p for cls, p in enumerate(pred)}
            cls = np.argmax(pred)
            if len(docids) == 1:
                (docid,) = docids
                soft_sentence_predictions = []
                start = 0
                for sent in documents[docid]:
                    end = start + len(sent)
                    soft_sentence_predictions.append(sum(attentions[start:end]))
                    start = end
                rat = {
                    'annotation_id': ann_id,
                    'classification': class_deinterner[cls],
                    'classification_scores': classification_scores,
                    'rationales': [{
                        'docid': docid,
                        'soft_rationale_predictions': attentions,
                        'soft_sentence_predictions': soft_sentence_predictions,
                    }]
                }
            else:
                raise ValueError()
            # TODO make hard predictions
            top_k = nlargest(int(k_fraction * len(doc)), zip(attentions, doc))
            top_k = [x[1] for x in top_k]
            evidence_only_docs.append(top_k)
            bottom_k = nsmallest(int((1 - k_fraction) * len(doc)), zip(attentions, doc))
            bottom_k = [x[1] for x in bottom_k]
            no_evidence_docs.append(bottom_k)
            rats.append(rat)

        for i, ((ann_id, _), pred, _) in enumerate(zip(ids, *decode_set(queries, evidence_only_docs))):
            classification_scores = {class_deinterner[cls]: p for cls, p in enumerate(pred)}
            rats[i]['sufficiency_classification_scores'] = classification_scores

        for i, ((ann_id, _), pred, _) in enumerate(zip(ids, *decode_set(queries, no_evidence_docs))):
            classification_scores = {class_deinterner[cls]: p for cls, p in enumerate(pred)}
            rats[i]['comprehensiveness_classification_scores'] = classification_scores

        return rats

    val_rats = generate_rationales(val_ids, val_queries, val_docs)
    test_rats = generate_rationales(test_ids, test_queries, test_docs)
    return val_rats, test_rats


def main():
    parser = argparse.ArgumentParser(description="""    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--model_params', dest='model_params', required=True,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    args = parser.parse_args()
    BATCH_FIRST = True

    with open(args.model_params, 'r') as fp:
        logging.debug(f'Loading model parameters from {args.model_params}')
        model_params = json.load(fp)
    train, val, test = load_datasets(args.data_dir)
    documents = load_documents(args.data_dir)
    document_vocab = set(chain.from_iterable(chain.from_iterable(documents.values())))
    annotation_vocab = set(chain.from_iterable(e.query.split() for e in chain(train, val, test)))
    logging.debug(f'Loaded {len(documents)} documents with {len(document_vocab)} unique words')
    # this ignores the case where annotations don't align perfectly with token boundaries, but this isn't that important
    vocab = document_vocab | annotation_vocab
    unk_token = 'UNK'
    classifier, word_interner, de_interner, evidence_classes = initialize_model(model_params, vocab,
                                                                                batch_first=BATCH_FIRST,
                                                                                unk_token=unk_token)
    classifier = classifier.cuda()
    logging.debug(
        f'Including annotations, we have {len(vocab)} total words in the data, with embeddings for {len(word_interner)}')
    interned_documents = intern_documents(documents, word_interner, unk_token)
    interned_train = intern_annotations(train, word_interner, unk_token)
    interned_val = intern_annotations(val, word_interner, unk_token)
    interned_test = intern_annotations(test, word_interner, unk_token)

    classifier, results = train_classifier(classifier, args.output_dir, interned_train, interned_val,
                                           interned_documents, model_params, evidence_classes)
    val_rats, test_rats = decode(classifier,
                                 interned_train,
                                 interned_val,
                                 interned_test,
                                 interned_documents,
                                 evidence_classes,
                                 batch_size=model_params['classifier']['batch_size'],
                                 tensorize_model_inputs=True,
                                 threshold=model_params['classifier']['threshold'],
                                 k_fraction=model_params['classifier']['k_fraction'],
                                 has_query=bool(model_params['classifier']['has_query']))
    with open(os.path.join(args.output_dir, 'val_decoded.jsonl'), 'w') as val_output:
        for line in val_rats:
            val_output.write(json.dumps(line))
            val_output.write('\n')

    with open(os.path.join(args.output_dir, 'test_decoded.jsonl'), 'w') as test_output:
        for line in test_rats:
            test_output.write(json.dumps(line))
            test_output.write('\n')
    #training_results, train_decoded, val_decoded, test_decoded = decode(classifier, interned_train, interned_val, interned_test, interned_documents, evidence_classes, params['classifier']['batch_size'], tensorize_model_inputs=True)
    #write_jsonl(train_decoded, os.path.join(args.output_dir, 'train_decoded.jsonl'))
    #write_jsonl(val_decoded, os.path.join(args.output_dir, 'val_decoded.jsonl'))
    #write_jsonl(test_decoded, os.path.join(args.output_dir, 'test_decoded.jsonl'))
    #with open(os.path.join(args.output_dir, 'identifier_results.json'), 'w') as ident_output, \
    #    open(os.path.join(args.output_dir, 'classifier_results.json'), 'w') as class_output:
    #    ident_output.write(json.dumps(evidence_ident_results))
    #    class_output.write(json.dumps(evidence_class_results))
    #for k, v in pipeline_results.items():
    #    if type(v) is dict:
    #        for k1, v1 in v.items():
    #            logging.info(f'Pipeline results for {k}, {k1}={v1}')
    #    else:
    #        logging.info(f'Pipeline results {k}\t={v}')


if __name__ == '__main__':
    main()

import logging
import os
import random

from collections import OrderedDict
from itertools import chain
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score

from rationale_benchmark.utils import Annotation

from rationale_benchmark.models.pipeline.pipeline_utils import (
    SentenceEvidence,
    annotations_to_evidence_identification,
    make_preds_epoch,
    score_rationales,
)


def _get_sampling_method(training_pars: dict) -> Callable[
    [List[SentenceEvidence], Dict[str, List[SentenceEvidence]]], List[SentenceEvidence]]:
    """Generates a sampler that produces (positive, negative) sentence-level examples

    Returns a function that takes a document converted to sentence level
    annotations and a dictionary of docid -> sentence level annotations, and
    returns a set of sentence level annotations.

    This sampling method is necessary as we can have far too many negative
    examples in our training data (almost nothing is actually evidence).

    n.b. this factory is clearly crying for modularization, again into
    something that would call for dependency injection, but for the duration of
    this project, this will be fine.
    """

    # TODO implement sampling for nearby sentences (within the document)
    if training_pars['sampling_method'] == 'random':
        sampling_ratio = training_pars['sampling_ratio']
        logging.info(f'Setting up random sampling with negative/positive ratio = {sampling_ratio}')

        def random_sampler(document: List[SentenceEvidence], _: Dict[str, List[SentenceEvidence]]) -> \
                List[SentenceEvidence]:
            """Takes all the positives from a document, and a random choice over negatives"""
            positives = list(filter(lambda s: s.kls == 1 and len(s.sentence) > 0, document))
            if any(map(lambda s: len(s.sentence) == 0, positives)):
                raise ValueError("Some positive sentences are of zero length!")
            all_negatives = list(filter(lambda s: s.kls == 0 and len(s.sentence) > 0, document))
            # handle an edge case where a document can be only or mostly evidence for a statement
            num_negatives = min(len(all_negatives), round(len(positives) * sampling_ratio))
            random_negatives = random.choices(all_negatives, k=num_negatives)
            # sort the results so the next step is deterministic,
            results = sorted(positives + random_negatives)
            # this is an inplace shuffle.
            random.shuffle(results)
            return results

        return random_sampler
    elif training_pars['sampling_method'] == 'everything':
        def everything_sampler(document: List[SentenceEvidence],
                               _: Dict[str, List[SentenceEvidence]]) -> List[SentenceEvidence]:
            return document
        return everything_sampler
    else:
        raise ValueError(f"Unknown sampling method for training: {training_pars['sampling_method']}")


def train_evidence_identifier(evidence_identifier: nn.Module,
                              save_dir: str,
                              train: List[Annotation],
                              val: List[Annotation],
                              documents: Dict[str, List[List[int]]],
                              model_pars: dict,
                              optimizer=None,
                              scheduler=None,
                              tensorize_model_inputs: bool = True) -> Tuple[nn.Module, dict]:
    """Trains a module for rationale identification.

    This method tracks loss on the entire validation set, saves intermediate
    models, and supports restoring from an unfinished state. The best model on
    the validation set is maintained, and the model stops training if a patience
    (see below) number of epochs with no improvement is exceeded.

    As there are likely too many negative examples to reasonably train a
    classifier on everything, every epoch we subsample the negatives.

    Args:
        evidence_identifier: a module like the AttentiveClassifier
        save_dir: a place to save intermediate and final results and models.
        train: a List of interned Annotation objects.
        val: a List of interned Annotation objects.
        documents: a Dict of interned sentences
        model_pars: Arbitrary parameters directory, assumed to contain an "evidence_identifier" sub-dict with:
            lr: learning rate
            batch_size: an int
            sampling_method: a string, plus additional params in the dict to define creation of a sampler
            epochs: the number of epochs to train for
            patience: how long to wait for an improvement before giving up.
            max_grad_norm: optional, clip gradients.
        optimizer: what pytorch optimizer to use, if none, initialize Adam
        scheduler: optional, do we want a scheduler involved in learning?
        tensorize_model_inputs: should we convert our data to tensors before passing it to the model?
                                Useful if we have a model that performs its own tokenization (e.g. BERT as a Service)

    Returns:
        the trained evidence identifier and a dictionary of intermediate results.
    """

    def _prep_data_for_epoch(evidence_data: Dict[str, Dict[str, List[SentenceEvidence]]],
                             sampler: Callable[
                                 [List[SentenceEvidence], Dict[str, List[SentenceEvidence]]], List[SentenceEvidence]]
                             ) -> List[SentenceEvidence]:
        output_sentences = []
        ann_ids = sorted(evidence_data.keys())
        # in place shuffle so we get a different per-epoch ordering
        random.shuffle(ann_ids)
        for ann_id in ann_ids:
            for docid, sentences in evidence_data[ann_id].items():
                data = sampler(sentences, None)
                output_sentences.extend(data)
        return output_sentences

    logging.info(f'Beginning training with {len(train)} annotations, {len(val)} for validation')
    evidence_identifier_output_dir = os.path.join(save_dir, 'evidence_identifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(evidence_identifier_output_dir, exist_ok=True)

    model_save_file = os.path.join(evidence_identifier_output_dir, 'evidence_identifier.pt')
    epoch_save_file = os.path.join(evidence_identifier_output_dir, 'evidence_identifier_epoch_data.pt')

    if optimizer is None:
        optimizer = torch.optim.Adam(evidence_identifier.parameters(), lr=model_pars['evidence_identifier']['lr'])
    criterion = nn.CrossEntropyLoss(reduction='none')
    sampling_method = _get_sampling_method(model_pars['evidence_identifier'])
    batch_size = model_pars['evidence_identifier']['batch_size']
    epochs = model_pars['evidence_identifier']['epochs']
    patience = model_pars['evidence_identifier']['patience']
    max_grad_norm = model_pars['evidence_classifier'].get('max_grad_norm', None)

    evidence_train_data = annotations_to_evidence_identification(train, documents)
    evidence_val_data = annotations_to_evidence_identification(val, documents)

    device = next(evidence_identifier.parameters()).device

    results = {
        # "sampled" losses do not represent the true data distribution, but do represent training data
        'sampled_epoch_train_losses': [],
        'sampled_epoch_val_losses': [],
        # "full" losses do represent the true data distribution
        'full_epoch_val_losses': [],
        'full_epoch_val_acc': [],
        'full_epoch_val_rationale_scores': [],
    }
    # allow restoring an existing training run
    start_epoch = 0
    best_epoch = -1
    best_val_loss = float('inf')
    best_model_state_dict = None
    epoch_data = {}
    if os.path.exists(epoch_save_file):
        evidence_identifier.load_state_dict(torch.load(model_save_file))
        epoch_data = torch.load(epoch_save_file)
        start_epoch = epoch_data['epoch'] + 1
        # handle finishing because patience was exceeded or we didn't get the best final epoch
        if bool(epoch_data.get('done', 0)):
            start_epoch = epochs
        results = epoch_data['results']
        best_epoch = start_epoch
        best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_identifier.state_dict().items()})
    logging.info(f'Training evidence identifier from epoch {start_epoch} until epoch {epochs}')
    optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):
        epoch_train_data = _prep_data_for_epoch(evidence_train_data, sampling_method)
        epoch_val_data = _prep_data_for_epoch(evidence_val_data, sampling_method)
        sampled_epoch_train_loss = 0
        evidence_identifier.train()
        logging.info(
            f'Training with {len(epoch_train_data) // batch_size} batches with {len(epoch_train_data)} examples')
        for batch_start in range(0, len(epoch_train_data), batch_size):
            batch_elements = epoch_train_data[batch_start:min(batch_start + batch_size, len(epoch_train_data))]
            # we sample every time to thereoretically get a better representation of instances over the corpus.
            # this might just take more time than doing so in advance.
            targets, queries, sentences = zip(*[(s.kls, s.query, s.sentence) for s in batch_elements])
            ids = [(s.ann_id, s.docid, s.index) for s in batch_elements]
            targets = torch.tensor(targets, dtype=torch.long, device=device)
            if tensorize_model_inputs:
                queries = [torch.tensor(q, dtype=torch.long) for q in queries]
                sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]
            preds = evidence_identifier(queries, ids, sentences)
            loss = criterion(preds, targets.to(device=preds.device)).sum()
            sampled_epoch_train_loss += loss.item()
            loss = loss / len(preds)
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(evidence_identifier.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        sampled_epoch_train_loss /= len(epoch_train_data)
        results['sampled_epoch_train_losses'].append(sampled_epoch_train_loss)
        logging.info(f'Epoch {epoch} sampled training loss {sampled_epoch_train_loss}')

        with torch.no_grad():
            evidence_identifier.eval()
            sampled_epoch_val_loss, _, sampled_epoch_val_hard_pred, sampled_epoch_val_truth = \
                make_preds_epoch(evidence_identifier,
                                 epoch_val_data,
                                 batch_size,
                                 device,
                                 criterion,
                                 tensorize_model_inputs)
            results['sampled_epoch_val_losses'].append(sampled_epoch_val_loss)
            sampled_epoch_val_acc = accuracy_score(sampled_epoch_val_truth, sampled_epoch_val_hard_pred)
            logging.info(f'Epoch {epoch} sampled val loss {sampled_epoch_val_loss}, acc {sampled_epoch_val_acc}')
            # evaluate over *all* of the validation data
            all_val_data = list(filter(lambda se: len(se.sentence) > 0, chain.from_iterable(
                chain.from_iterable(x.values() for x in evidence_val_data.values()))))
            epoch_val_loss, epoch_val_soft_pred, epoch_val_hard_pred, epoch_val_truth = \
                make_preds_epoch(evidence_identifier,
                                 all_val_data,
                                 batch_size,
                                 device,
                                 criterion,
                                 tensorize_model_inputs)
            results['full_epoch_val_losses'].append(epoch_val_loss)
            results['full_epoch_val_acc'].append(accuracy_score(epoch_val_truth, epoch_val_hard_pred))
            results['full_epoch_val_rationale_scores'].append(
                score_rationales(val, documents, epoch_val_data, epoch_val_soft_pred))
            logging.info(
                f'Epoch {epoch} full val loss {epoch_val_loss}, accuracy: {results["full_epoch_val_acc"][-1]}, rationale scores: {results["full_epoch_val_rationale_scores"][-1]}')

            # if epoch_val_loss < best_val_loss:
            if sampled_epoch_val_loss < best_val_loss:
                logging.debug(f'Epoch {epoch} new best model with sampled val loss {sampled_epoch_val_loss}')
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_identifier.state_dict().items()})
                best_epoch = epoch
                best_val_loss = sampled_epoch_val_loss
                torch.save(evidence_identifier.state_dict(), model_save_file)
                epoch_data = {
                    'epoch': epoch,
                    'results': results,
                    'best_val_loss': best_val_loss,
                    'done': 0
                }
                torch.save(epoch_data, epoch_save_file)
        if epoch - best_epoch > patience:
            epoch_data['done'] = 1
            torch.save(epoch_data, epoch_save_file)
            break

    epoch_data['done'] = 1
    epoch_data['results'] = results
    torch.save(epoch_data, epoch_save_file)
    evidence_identifier.load_state_dict(best_model_state_dict)
    evidence_identifier = evidence_identifier.to(device=device)
    evidence_identifier.eval()
    return evidence_identifier, results

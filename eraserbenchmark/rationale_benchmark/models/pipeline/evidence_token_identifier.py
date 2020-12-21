import itertools
import logging
import os
import random

from collections import OrderedDict
from itertools import chain
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict, Tuple, Callable

import torch

from torch import nn

from rationale_benchmark.models.pipeline.pipeline_utils import SentenceEvidence, score_rationales, \
    annotations_to_evidence_token_identification, make_token_preds_epoch
from rationale_benchmark.utils import Annotation
from rationale_benchmark.models.model_utils import PaddedSequence


def _get_sampling_method(training_pars: dict) -> Callable[
    [List[SentenceEvidence], Dict[str, List[SentenceEvidence]]], List[SentenceEvidence]]:
    """Generates a sampler that produces sentences with a mix of positive and negative evidence tokens

    Returns a function that takes a document converted to sentence level
    annotations and a dictionary of docid -> sentence level annotations, and
    returns a set of sentence level annotations.

    This is theoretically necessary to support a pipeline with three stages: evidence sentence identification, token
    identification, followed by classification on the survivors.

    For e-SNLI and COS-E the correct assignment is to use "everything" or a variant like everything.
    """

    if training_pars['sampling_method'] == 'everything':
        def everything_sampler(document: List[SentenceEvidence],
                               _: Dict[str, List[SentenceEvidence]]) -> List[SentenceEvidence]:
            assert len(document) == 1
            return document[0]
        return everything_sampler
    else:
        raise ValueError(f"Unknown sampling method for training: {training_pars['sampling_method']}")


def train_evidence_token_identifier(evidence_token_identifier: nn.Module,
                                    save_dir: str,
                                    train: List[Annotation],
                                    val: List[Annotation],
                                    interned_documents: Dict[str, List[List[int]]],
                                    source_documents: Dict[str, List[List[str]]],
                                    token_mapping: Dict[str, List[List[Tuple[int, int]]]],
                                    model_pars: dict,
                                    optimizer=None,
                                    scheduler=None,
                                    tensorize_model_inputs: bool = True) -> Tuple[nn.Module, dict]:
    """Trains a module for token-level rationale identification.

    This method tracks loss on the entire validation set, saves intermediate
    models, and supports restoring from an unfinished state. The best model on
    the validation set is maintained, and the model stops training if a patience
    (see below) number of epochs with no improvement is exceeded.

    As there are likely too many negative examples to reasonably train a
    classifier on everything, every epoch we subsample the negatives.

    Args:
        evidence_token_identifier: a module like the AttentiveClassifier
        save_dir: a place to save intermediate and final results and models.
        train: a List of interned Annotation objects.
        val: a List of interned Annotation objects.
        interned_documents: a Dict of interned sentences
        source_documents: 
        token_mapping:
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
        the trained evidence token identifier and a dictionary of intermediate results.
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
                output_sentences.append(data)
        return output_sentences

    logging.info(f'Beginning training with {len(train)} annotations, {len(val)} for validation')
    evidence_identifier_output_dir = os.path.join(save_dir, 'evidence_token_identifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(evidence_identifier_output_dir, exist_ok=True)

    model_save_file = os.path.join(evidence_identifier_output_dir, 'evidence_token_identifier.pt')
    epoch_save_file = os.path.join(evidence_identifier_output_dir, 'evidence_token_identifier_epoch_data.pt')

    if optimizer is None:
        optimizer = torch.optim.Adam(evidence_token_identifier.parameters(), lr=model_pars['evidence_token_identifier']['lr'])
    criterion = nn.BCELoss(reduction='none')
    sampling_method = _get_sampling_method(model_pars['evidence_token_identifier'])
    batch_size = model_pars['evidence_token_identifier']['batch_size']
    epochs = model_pars['evidence_token_identifier']['epochs']
    patience = model_pars['evidence_token_identifier']['patience']
    max_grad_norm = model_pars['evidence_token_identifier'].get('max_grad_norm', None)
    use_cose_hack = bool(model_pars['evidence_token_identifier'].get('cose_data_hack', 0))

    # annotation id -> docid -> [SentenceEvidence])
    evidence_train_data = annotations_to_evidence_token_identification(train,
                                                                       source_documents=source_documents,
                                                                       interned_documents=interned_documents,
                                                                       token_mapping=token_mapping)
    evidence_val_data = annotations_to_evidence_token_identification(val,
                                                                     source_documents=source_documents,
                                                                     interned_documents=interned_documents,
                                                                     token_mapping=token_mapping)

    device = next(evidence_token_identifier.parameters()).device

    results = {
        # "sampled" losses do not represent the true data distribution, but do represent training data
        'sampled_epoch_train_losses': [],
        # "full" losses do represent the true data distribution
        'epoch_val_losses': [],
        'epoch_val_acc': [],
        'epoch_val_f': [],
        'epoch_val_rationale_scores': [],
    }
    # allow restoring an existing training run
    start_epoch = 0
    best_epoch = -1
    best_val_loss = float('inf')
    best_model_state_dict = None
    epoch_data = {}
    if os.path.exists(epoch_save_file):
        evidence_token_identifier.load_state_dict(torch.load(model_save_file))
        epoch_data = torch.load(epoch_save_file)
        start_epoch = epoch_data['epoch'] + 1
        # handle finishing because patience was exceeded or we didn't get the best final epoch
        if bool(epoch_data.get('done', 0)):
            start_epoch = epochs
        results = epoch_data['results']
        best_epoch = start_epoch
        best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_token_identifier.state_dict().items()})
    logging.info(f'Training evidence identifier from epoch {start_epoch} until epoch {epochs}')
    optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):
        epoch_train_data = _prep_data_for_epoch(evidence_train_data, sampling_method)
        epoch_val_data = _prep_data_for_epoch(evidence_val_data, sampling_method)
        sampled_epoch_train_loss = 0
        evidence_token_identifier.train()
        logging.info(
            f'Training with {len(epoch_train_data) // batch_size} batches with {len(epoch_train_data)} examples')
        for batch_start in range(0, len(epoch_train_data), batch_size):
            batch_elements = epoch_train_data[batch_start:min(batch_start + batch_size, len(epoch_train_data))]
            # we sample every time to thereoretically get a better representation of instances over the corpus.
            # this might just take more time than doing so in advance.
            targets, queries, sentences = zip(*[(s.kls, s.query, s.sentence) for s in batch_elements])
            ids = [(s.ann_id, s.docid, s.index) for s in batch_elements]
            #targets = torch.tensor(targets, dtype=torch.long, device=device)
            targets = PaddedSequence.autopad([torch.tensor(t, dtype=torch.long, device=device) for t in targets], batch_first=True, device=device)
            aggregate_spans = [token_mapping[s.docid][s.index] for s in batch_elements]
            if tensorize_model_inputs:
                if all(q is None for q in queries):
                    queries = [torch.tensor([], dtype=torch.long) for _ in queries]
                else:
                    assert all(q is not None for q in queries)
                    queries = [torch.tensor(q, dtype=torch.long) for q in queries]
                sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]
            preds = evidence_token_identifier(queries, ids, sentences, aggregate_spans)
            mask = targets.mask(on=1, off=0, device=device, dtype=torch.float)
            preds = preds * mask
            loss = criterion(preds, (targets.data.to(device=preds.device) * mask).squeeze()).sum()
            sampled_epoch_train_loss += loss.item()
            loss = loss / torch.sum(mask)
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(evidence_token_identifier.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        sampled_epoch_train_loss /= len(epoch_train_data)
        results['sampled_epoch_train_losses'].append(sampled_epoch_train_loss)
        logging.info(f'Epoch {epoch} training loss {sampled_epoch_train_loss}')

        with torch.no_grad():
            evidence_token_identifier.eval()
            epoch_val_loss, epoch_val_soft_pred, epoch_val_hard_pred, epoch_val_truth = \
                make_token_preds_epoch(evidence_token_identifier,
                                       epoch_val_data,
                                       token_mapping,
                                       batch_size,
                                       device,
                                       criterion,
                                       tensorize_model_inputs)
            #epoch_val_soft_pred = list(chain.from_iterable(epoch_val_soft_pred))
            epoch_val_hard_pred = list(chain.from_iterable(epoch_val_hard_pred))
            epoch_val_truth = list(chain.from_iterable(epoch_val_truth))
            results['epoch_val_losses'].append(epoch_val_loss)
            results['epoch_val_acc'].append(accuracy_score(epoch_val_truth, epoch_val_hard_pred))
            results['epoch_val_f'].append(classification_report(epoch_val_truth, epoch_val_hard_pred, output_dict=True))
            epoch_val_soft_pred_for_scoring = [[[1 - z, z] for z in y] for y in epoch_val_soft_pred]
            logging.info(
                f'Epoch {epoch} full val loss {epoch_val_loss}, accuracy: {results["epoch_val_acc"][-1]}, f: {results["epoch_val_f"][-1]}, rationale scores: look, it\'s already a pain to duplicate this code. What do you want from me.')

            # if epoch_val_loss < best_val_loss:
            if epoch_val_loss < best_val_loss:
                logging.debug(f'Epoch {epoch} new best model with val loss {epoch_val_loss}')
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_token_identifier.state_dict().items()})
                best_epoch = epoch
                best_val_loss = epoch_val_loss
                torch.save(evidence_token_identifier.state_dict(), model_save_file)
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
    evidence_token_identifier.load_state_dict(best_model_state_dict)
    evidence_token_identifier = evidence_token_identifier.to(device=device)
    evidence_token_identifier.eval()
    return evidence_token_identifier, results

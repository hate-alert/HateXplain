import logging
import os
import random

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, classification_report

from rationale_benchmark.utils import Annotation

from rationale_benchmark.models.pipeline.pipeline_utils import (
    annotations_to_evidence_classification,
    token_annotations_to_evidence_classification,
    make_preds_epoch,
)


def train_evidence_classifier(evidence_classifier: nn.Module,
                              save_dir: str,
                              train: List[Annotation],
                              val: List[Annotation],
                              documents: Dict[str, List[List[int]]],
                              model_pars: dict,
                              class_interner: Dict[str, int],
                              optimizer=None,
                              scheduler=None,
                              tensorize_model_inputs: bool = True,
                              token_only_evidence: bool=False) -> Tuple[nn.Module, dict]:
    """Trains an end-task classifier based on the ground truth evidence

    This method tracks loss on the validation set, saves intermediate
    models, and supports restoring from an unfinished state. The best model on
    the validation set is maintained, and the model stops training if a patience
    (see below) number of epochs with no improvement is exceeded.

    Args:
        evidence_classifier: a module like the AttentiveClassifier
        save_dir: a place to save intermediate and final results and models.
        train: a List of interned Annotation objects.
        val: a List of interned Annotation objects.
        documents: a Dict of interned sentences
        model_pars: Arbitrary parameters directory, assumed to contain an "evidence_classifier" sub-dict with:
            lr: learning rate
            batch_size: an int
            sampling_method: a string, plus additional params in the dict to define creation of a sampler.
                This should probably just be "everything"
            epochs: the number of epochs to train for
            patience: how long to wait for an improvement before giving up.
            max_grad_norm: optional, clip gradients.
        class_interner: an object for converting Annotation classes into ints.
        optimizer: what pytorch optimizer to use, if none, initialize Adam
        scheduler: optional, do we want a scheduler involved in learning?
        tensorize_model_inputs: should we convert our data to tensors before passing it to the model?
                                Useful if we have a model that performs its own tokenization (e.g. BERT as a Service)

    Returns:
        the trained evidence classifier and a dictionary of intermediate results.
    """
    logging.info(f'Beginning training evidence classifier with {len(train)} annotations, {len(val)} for validation')
    evidence_classifier_output_dir = os.path.join(save_dir, 'evidence_classifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(evidence_classifier_output_dir, exist_ok=True)
    model_save_file = os.path.join(evidence_classifier_output_dir, 'evidence_classifier.pt')
    epoch_save_file = os.path.join(evidence_classifier_output_dir, 'evidence_classifier_epoch_data.pt')

    device = next(evidence_classifier.parameters()).device
    if optimizer is None:
        optimizer = torch.optim.Adam(evidence_classifier.parameters(), lr=model_pars['evidence_classifier']['lr'])
    criterion = nn.CrossEntropyLoss(reduction='none')
    batch_size = model_pars['evidence_classifier']['batch_size']
    epochs = model_pars['evidence_classifier']['epochs']
    patience = model_pars['evidence_classifier']['patience']
    max_grad_norm = model_pars['evidence_classifier'].get('max_grad_norm', None)

    if token_only_evidence:
        evidence_train_data = token_annotations_to_evidence_classification(train, documents, class_interner)
        evidence_val_data = token_annotations_to_evidence_classification(val, documents, class_interner)
    else:
        evidence_train_data = annotations_to_evidence_classification(train, documents, class_interner, include_all=False)
        evidence_val_data = annotations_to_evidence_classification(val, documents, class_interner, include_all=False)

    class_labels = [k for k, v in sorted(class_interner.items())]

    results = {
        'train_loss': [],
        'train_f1': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': [],
    }
    best_epoch = -1
    best_val_loss = float('inf')
    best_model_state_dict = None
    start_epoch = 0
    epoch_data = {}
    if os.path.exists(epoch_save_file):
        logging.info(f'Restoring model from {model_save_file}')
        evidence_classifier.load_state_dict(torch.load(model_save_file))
        epoch_data = torch.load(epoch_save_file)
        start_epoch = epoch_data['epoch'] + 1
        # handle finishing because patience was exceeded or we didn't get the best final epoch
        if bool(epoch_data.get('done', 0)):
            start_epoch = epochs
        results = epoch_data['results']
        best_epoch = start_epoch
        best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_classifier.state_dict().items()})
        logging.info(f'Restoring training from epoch {start_epoch}')
    logging.info(f'Training evidence classifier from epoch {start_epoch} until epoch {epochs}')
    optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):
        epoch_train_data = random.sample(evidence_train_data, k=len(evidence_train_data))
        epoch_val_data = random.sample(evidence_val_data, k=len(evidence_val_data))
        epoch_train_loss = 0
        evidence_classifier.train()
        logging.info(
            f'Training with {len(epoch_train_data) // batch_size} batches with {len(epoch_train_data)} examples')
        for batch_start in range(0, len(epoch_train_data), batch_size):
            batch_elements = epoch_train_data[batch_start:min(batch_start + batch_size, len(epoch_train_data))]
            targets, queries, sentences = zip(*[(s.kls, s.query, s.sentence) for s in batch_elements])
            ids = [(s.ann_id, s.docid, s.index) for s in batch_elements]
            targets = torch.tensor(targets, dtype=torch.long, device=device)
            if tensorize_model_inputs:
                queries = [torch.tensor(q, dtype=torch.long) for q in queries]
                sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]
            preds = evidence_classifier(queries, ids, sentences)
            loss = criterion(preds, targets.to(device=preds.device)).sum()
            epoch_train_loss += loss.item()
            loss = loss / len(preds)  # accumulate entire loss above
            loss.backward()
            assert loss == loss  # for nans
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(evidence_classifier.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        epoch_train_loss /= len(epoch_train_data)
        assert epoch_train_loss == epoch_train_loss  # for nans
        results['train_loss'].append(epoch_train_loss)
        logging.info(f'Epoch {epoch} training loss {epoch_train_loss}')

        with torch.no_grad():
            epoch_train_loss, epoch_train_soft_pred, epoch_train_hard_pred, epoch_train_truth = make_preds_epoch(
                evidence_classifier, epoch_train_data, batch_size, device, criterion=criterion,
                tensorize_model_inputs=tensorize_model_inputs)
            results['train_f1'].append(
                classification_report(epoch_train_truth, epoch_train_hard_pred, target_names=class_labels,
                                      output_dict=True))
            results['train_acc'].append(accuracy_score(epoch_train_truth, epoch_train_hard_pred))
            epoch_val_loss, epoch_val_soft_pred, epoch_val_hard_pred, epoch_val_truth = make_preds_epoch(
                evidence_classifier, epoch_val_data, batch_size, device, criterion=criterion,
                tensorize_model_inputs=tensorize_model_inputs)
            results['val_loss'].append(epoch_val_loss)
            results['val_f1'].append(
                classification_report(epoch_val_truth, epoch_val_hard_pred, target_names=class_labels,
                                      output_dict=True))
            results['val_acc'].append(accuracy_score(epoch_val_truth, epoch_val_hard_pred))
            assert epoch_val_loss == epoch_val_loss  # for nans
            logging.info(f'Epoch {epoch} val loss {epoch_val_loss}')
            logging.info(f'Epoch {epoch} val acc {results["val_acc"][-1]}')
            logging.info(f'Epoch {epoch} val f1 {results["val_f1"][-1]}')

            if epoch_val_loss < best_val_loss:
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_classifier.state_dict().items()})
                best_epoch = epoch
                best_val_loss = epoch_val_loss
                epoch_data = {
                    'epoch': epoch,
                    'results': results,
                    'best_val_loss': best_val_loss,
                    'done': 0,
                }
                torch.save(evidence_classifier.state_dict(), model_save_file)
                torch.save(epoch_data, epoch_save_file)
                logging.debug(f'Epoch {epoch} new best model with val loss {epoch_val_loss}')
        if epoch - best_epoch > patience:
            logging.info(f'Exiting after epoch {epoch} due to no improvement')
            epoch_data['done'] = 1
            torch.save(epoch_data, epoch_save_file)
            break

    epoch_data['done'] = 1
    epoch_data['results'] = results
    torch.save(epoch_data, epoch_save_file)
    evidence_classifier.load_state_dict(best_model_state_dict)
    evidence_classifier = evidence_classifier.to(device=device)
    evidence_classifier.eval()
    return evidence_classifier, results

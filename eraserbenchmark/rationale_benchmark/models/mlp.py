from typing import Any, List

import torch
import torch.nn as nn

from transformers import BertForSequenceClassification

from rationale_benchmark.models.model_utils import PaddedSequence


class WordEmbedder(nn.Module):
    """ A thin wrapping for an nn.embedding """

    def __init__(self, embeddings: nn.Embedding, dropout_rate: float):
        super(WordEmbedder, self).__init__()
        self.embeddings = embeddings
        self.output_dimension = embeddings.embedding_dim
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, docids: List[Any], ps: PaddedSequence) -> PaddedSequence:
        # n.b. the docids are to allow for wrapping a pre-decoded BERT or ELMo or $FAVORITE_LANGUAGE_MODEL_OF_THE_DAY
        if docids and len(docids) not in ps.data.size():
            raise ValueError(f"Document id dimension {len(docids)} does not match input data dimensions {ps.data.size()}")
        embedded = self.embeddings(ps.data)
        embedded = self.dropout(embedded)
        return PaddedSequence(embedded, ps.batch_sizes, ps.batch_first)


class LuongAttention(nn.Module):
    def __init__(self,
                 output_size: int,
                 query_size: int=None,
                 dropout_rate:float=0.0):
        super(LuongAttention, self).__init__()
        self.use_query = query_size is not None
        self.query_size = query_size
        self.hidden_size = output_size
        input_size = query_size + output_size if self.use_query else output_size
        self.w = nn.Linear(input_size, output_size)
        self.v = nn.Parameter(torch.randn((output_size,)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,
                output: PaddedSequence, # batch x length x rep
                query: torch.Tensor=None):
        assert output.batch_first
        if query is not None:
            query = query.unsqueeze(1).repeat((1, output.data.size()[1], 1))
            attn_input = torch.cat([output.data, query], dim=2)
        else:
            attn_input = output.data
        raw_score = self.w(attn_input)
        score = torch.tanh(raw_score) @ self.v
        score = score + output.mask(size=score.data.size(),
                                    on=0,
                                    off=float('-inf'),
                                    dtype=torch.float,
                                    device=self.v.device)
        weights = torch.softmax(score, dim=1).unsqueeze(dim=-1)
        expectation = weights * output.data
        expectation = expectation.sum(dim=1)
        expectation = self.dropout(expectation)
        return score, weights, expectation


class BahadanauAttention(nn.Module):
    
    def __init__(self,
                 output_size: int,
                 query_size: int=None,
                 dropout_rate:float=0.0):
        super(BahadanauAttention, self).__init__()
        self.v = nn.Parameter(torch.randn((output_size, 1)), requires_grad=True)
        self.w = nn.Linear(output_size, output_size, bias=False)
        if query_size:
            self.u = nn.Linear(query_size, output_size, bias=False)
        else:
            self.u = None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_dimension = output_size

    def forward(self,
                output: PaddedSequence, # batch x length x rep OR length x batch x rep
                query: torch.Tensor=None):
        raw_score = self.w(output.data)
        if self.u:
            raw_score += self.u(query.unsqueeze(1))
        score = torch.tanh(raw_score) @ self.v
        score = score + output.mask(
                                size=output.data.size()[:2],
                                on=0,
                                off=float('-inf'),
                                dtype=torch.float,
                                device=self.v.device).unsqueeze(dim=-1)
        dimension = 1 if output.batch_first else 0
        assert output.batch_first # for simplicity we don't bother
        weights = torch.softmax(score, dim=dimension)
        expectation = weights * output.data
        expectation = expectation.sum(dim=dimension)
        expectation = self.dropout(expectation)
        return score, weights, expectation


class RNNEncoder(nn.Module):
    """Recurrently encodes a sequence of words into a single vector.

    Collapsing the sequence of encoded values can be done either via an
    attention mechanism, or if none provided, by just using the final hidden
    states.
    """

    def __init__(self,
                 word_embedder: WordEmbedder,
                 output_dimension: int=None,
                 condition: bool=False,
                 batch_first: bool=False,
                 dropout_rate: float=0,
                 num_layers=1,
                 bidirectional=False,
                 attention_mechanism=None):
        super(RNNEncoder, self).__init__()
        if output_dimension is None:
            output_dimension = word_embedder.output_dimension
        self.word_embedder = word_embedder
        self.output_dimension = output_dimension
        self.condition = condition
        self.batch_first = batch_first
        input_size = word_embedder.output_dimension
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=self.output_dimension,
                          batch_first=batch_first,
                          dropout=dropout_rate,
                          num_layers=num_layers,
                          bidirectional=bidirectional)
        self.attention_mechanism = attention_mechanism

    def forward(self, 
                docids: List[Any],
                docs: PaddedSequence,
                query: torch.Tensor):
        embedded = self.word_embedder(docids, docs)
        assert embedded.batch_first == self.rnn.batch_first
        # concatenate the query to every input
        if self.condition:
            assert query is not None
            #if self.batch_first:
            #    query = torch.cat(docs.data.size()[1]*[query.unsqueeze(dim=1)],dim=1)
            #else:
            #    # TODO verify this works properly!
            #    import pdb; pdb.set_trace()
            #    query = torch.cat(docs.data.size()[0]*[query.unsqueeze(dim=0)],dim=0)
            #embedded = torch.cat([query, embedded.data], dim=-1)
        # this doesn't handle multilayer and multidirectional cases
        output, hidden = self.rnn(docs.pack_other(embedded.data))
        output = PaddedSequence.from_packed_sequence(output, batch_first=docs.batch_first)
        assert hidden.size()[-1] == self.rnn.hidden_size
        if self.attention_mechanism is not None:
            unnormalized_attention, attention, hidden = self.attention_mechanism(output, query)
            assert hidden.size()[-1] == self.rnn.hidden_size
        else:
            unnormalized_attention, attention, hidden = None, None, hidden
        return hidden, unnormalized_attention, attention, output


class AttentiveClassifier(nn.Module):
    """Encodes a document + a query and makes a classification. Supports query-only and document-only modes.

    Args:
      query_encoder:
        - takes a list of query ids, query representation, and optional additional encoding element to a fixed size.
        - same parameterization as the document_encoder (just think of the query as a document)
      document_encoder: 
        - takes a list of docids (for convenience if working with pre-computed representations), document representations, and an encoded query to a fixed size
      num_classes:
        - how many things to make a prediction for
      mlp_size:
        - document + query -> linear (mlp_size) -> non-linear -> num_classes -> softmax
    """

    def __init__(self, 
                 document_encoder: RNNEncoder,
                 query_encoder: RNNEncoder,
                 num_classes: int,
                 mlp_size: int,
                 dropout_rate: float):
        super(AttentiveClassifier, self).__init__()
        self.document_encoder = document_encoder
        self.query_encoder = query_encoder

        document_output_dimension = self.document_encoder.output_dimension if document_encoder else 0
        query_output_dimension = self.query_encoder.output_dimension if query_encoder else 0

        self.mlp = nn.Sequential(nn.Dropout(p=dropout_rate),
                                 nn.Linear(document_output_dimension + query_output_dimension, mlp_size),
                                 nn.ReLU(),
                                 nn.Dropout(p=dropout_rate),
                                 nn.Linear(mlp_size, num_classes),
                                 nn.Softmax(dim=-1))

    def forward(self,
                query: List[torch.tensor],
                docids: List[Any],
                document_batch: List[torch.tensor],
                return_attentions: bool=False):
        # note about device management:
        # since distributed training is enabled, the inputs to this module can be on *any* device (preferably cpu, since we wrap and unwrap the module)
        # we want to keep these params on the input device (assuming CPU) for as long as possible for cheap memory access
        device = next(self.parameters()).device
        if query is not None:
            assert self.query_encoder is not None
            query = PaddedSequence.autopad(query, batch_first=self.query_encoder.batch_first, device=device)
            query_vector, unnormalized_query_attention, normalized_query_attention, _ = self.query_encoder(None, query, None)
            unnormalized_query_attention = PaddedSequence(unnormalized_query_attention, query.batch_sizes, query.batch_first)
            normalized_query_attention = PaddedSequence(normalized_query_attention, query.batch_sizes, query.batch_first)
        else:
            query_vector = None
            unnormalized_query_attention = None
            normalized_query_attention = None
        if document_batch is not None:
            assert self.document_encoder is not None
            document_batch = PaddedSequence.autopad(document_batch, batch_first=self.document_encoder.batch_first, device=device)
            document_vector, unnormalized_document_attention, normalized_document_attention, _ = self.document_encoder(docids, document_batch, query_vector)
            unnormalized_document_attention = PaddedSequence(unnormalized_document_attention, document_batch.batch_sizes, document_batch.batch_first)
            normalized_document_attention = PaddedSequence(normalized_document_attention, document_batch.batch_sizes, document_batch.batch_first)
        else:
            document_vector = None
        if query_vector is not None and document_vector is not None:
            assert query_vector.size()[:2] == document_vector.size()[:2]
            combined_vector = torch.cat([query_vector, document_vector], dim=-1)
        else:
            assert query_vector is not None or document_vector is not None
            combined_vector = query_vector if query_vector is not None else document_vector
        if return_attentions:
            return self.mlp(combined_vector), unnormalized_query_attention, normalized_query_attention, unnormalized_document_attention, normalized_document_attention
        else:
            return self.mlp(combined_vector)


class BertClassifier(nn.Module):
    """Thin wrapper around BertForSequenceClassification"""
    def __init__(self,
                 bert_dir: str,
                 pad_token_id: int,
                 cls_token_id: int,
                 sep_token_id: int,
                 num_labels: int,
                 max_length: int=512,
                 use_half_precision=True):
        super(BertClassifier, self).__init__()
        bert = BertForSequenceClassification.from_pretrained(bert_dir, num_labels=num_labels)
        if use_half_precision:
            import apex
            bert = bert.half()
        self.bert = bert
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def forward(self,
                query: List[torch.tensor],
                docids: List[Any],
                document_batch: List[torch.tensor]):
        assert len(query) == len(document_batch)
        # note about device management:
        # since distributed training is enabled, the inputs to this module can be on *any* device (preferably cpu, since we wrap and unwrap the module)
        # we want to keep these params on the input device (assuming CPU) for as long as possible for cheap memory access
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id]).to(device=document_batch[0].device)
        sep_token = torch.tensor([self.sep_token_id]).to(device=document_batch[0].device)
        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[:(self.max_length - len(q) - 2)]
            input_tensors.append(torch.cat([cls_token, q, sep_token, d]))
            position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id, device=target_device)
        positions = PaddedSequence.autopad(position_ids, batch_first=True, padding_value=0, device=target_device)
        (classes,) = self.bert(bert_input.data, attention_mask=bert_input.mask(on=0.0, off=float('-inf'), device=target_device), position_ids=positions.data)
        assert torch.all(classes == classes) # for nans
        return classes

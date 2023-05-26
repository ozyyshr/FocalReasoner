# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """

import math
# import dgl
# from gcn import RGCNModel
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import random
from transformers import RobertaForMultipleChoice, RobertaModel

from transformers.activations import ACT2FN, gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers import RobertaConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


class TriLinear(nn.Module):

    def __init__(self, input_size):
        super(TriLinear, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(1, input_size))
        self.w2 = nn.Parameter(torch.FloatTensor(1, input_size))
        self.w3 = nn.Parameter(torch.FloatTensor(1, input_size))

        self.init_param()

    def forward(self, query, key):
        ndim = query.dim()
        q_logit = F.linear(query, self.w1)
        k_logit = F.linear(key, self.w2)

        shape = [1] * (ndim - 1) + [-1]
        dot_k = self.w3.view(shape) * key
        dot_logit = torch.matmul(query, torch.transpose(dot_k, -1, -2))

        logit = q_logit + torch.transpose(k_logit, -1, -2) + dot_logit
        return logit

    def init_param(self):
        init.normal_(self.w1, 0., 0.02)
        init.normal_(self.w2, 0., 0.02)
        init.normal_(self.w3, 0., 0.02)

class SCAttention(nn.Module) :
    def __init__(self, input_size, hidden_size) :
        super(SCAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size)
        self.map_linear = nn.Linear(hidden_size, hidden_size)
        #self.map_linear = nn.Linear(4 * hidden_size, hidden_size)
        #self.map_linear = nn.Linear(4 * hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.W.weight.data)
        self.W.bias.data.fill_(0.1)

    #W_wq_relu_cox4_nolastRelu
    def forward(self, passage, p_mask, question, q_mask):
        Wp = F.relu(self.W(passage))
        Wq = F.relu(self.W(question))
        scores = torch.bmm(Wp, Wq.transpose(2, 1))
        # mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)
        q_mask = torch.transpose(q_mask, 1, 2)
        mask = q_mask.repeat(1, passage.size(1), 1)
        alpha = masked_softmax(scores, mask)
        output = torch.bmm(alpha, Wq)
        output = self.map_linear(output)
        return output

class Attention(nn.Module):

    def __init__(self, sim):
        super(Attention, self).__init__()
        self.sim = sim

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        ndim = query.dim()
        logit = self.sim(query, key)
        if query_mask is not None and key_mask is not None:
            mask = query_mask.unsqueeze(ndim - 1) * key_mask.unsqueeze(ndim - 2)
            logit = logit.masked_fill(~mask, -float('inf'))

        attn_weight = F.softmax(logit, dim=-1)
        if query_mask is not None and key_mask is not None:
            attn_weight = attn_weight.masked_fill(~mask, 0.)

        attn = torch.matmul(attn_weight, value)

        kq_weight = F.softmax(logit, dim=1)
        if query_mask is not None and key_mask is not None:
            kq_weight = kq_weight.masked_fill(~mask, 0.)

        co_weight = torch.matmul(attn_weight, torch.transpose(kq_weight, -1, -2))
        co_attn = torch.matmul(co_weight, query)

        return (attn, attn_weight), (co_attn, co_weight)

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        #mask = mask.half()

        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

class AttentivePooling(nn.Module):

    def __init__(self, input_size):
        super(AttentivePooling, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, input, mask):
        bsz, length, size = input.size()
        score = self.fc(input.contiguous().view(-1, size)).view(bsz, length)
        score = score.masked_fill(~mask, -float('inf'))
        prob = F.softmax(score, dim=-1)
        attn = torch.bmm(prob.unsqueeze(1), input)
        return attn


# Copied from transformers.models.bert.modeling_bert.BertPooler
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


ROBERTA_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.RobertaTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

class CNN_conv1d(nn.Module):
    def __init__(self, config, filter_size=3):
        super(CNN_conv1d, self).__init__()
        self.char_dim = config.hidden_size
        self.filter_size = filter_size #max_word_length
        self.out_channels = self.char_dim
        self.char_cnn =nn.Conv1d(self.char_dim, self.char_dim,kernel_size=self.filter_size,
                     padding=0)
        self.relu = nn.ReLU()
        #print("dropout:",str(config.hidden_dropout_prob))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs, max_word_len):
        """
        Arguments:
            inputs: [batch_size, word_len, char_len]
        """
        if(len(inputs.size())>3):
            bsz, word_len,  max_word_len, dim = inputs.size()
            #print(bsz, word_len,  max_word_len, dim)
        else:
            bsz, word_len, dim = inputs.size()
            word_len = int(word_len / max_word_len)

        inputs = inputs.view(-1, max_word_len, dim)
        x = inputs.transpose(1, 2)
        x = self.char_cnn(x)
        x = self.relu(x)
        x = F.max_pool1d(x, kernel_size=x.size(-1))
        x = self.dropout(x.squeeze())

        return x.view(bsz, word_len, -1)

@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
# class RobertaForMultipleChoice(RobertaPreTrainedModel):
class RobertaForMultipleChoiceSVO(RobertaForMultipleChoice):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, max_doc_len, max_query_len, max_option_len, svo_weight):
        # super(RobertaForMultipleChoice, self).__init__(config)
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.num_labels = 4
        self.max_doc_len = max_doc_len
        self.max_query_len = max_query_len
        self.max_option_len = max_option_len
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.hidden_size = config.hidden_size
        self.score_fc = nn.Linear(config.hidden_size, 1)
        self.attn_sim = TriLinear(config.hidden_size)
        self.attention = Attention(sim=self.attn_sim)
        self.attn_fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=True)
        
        self.opt_attn_sim = TriLinear(config.hidden_size)
        self.comp_fc = nn.Linear(config.hidden_size*7, config.hidden_size, bias=True)
        self.opt_attention = Attention(sim=self.opt_attn_sim)
        # self.GCN = RGCNModel(config.hidden_size, 5, 1, True)
        # self.GCN.to(self.device)
        
        self.opt_selfattn_sim = TriLinear(config.hidden_size)
        self.opt_self_attention = Attention(sim=self.opt_selfattn_sim)
        self.opt_selfattn_fc = nn.Linear(config.hidden_size * 4, config.hidden_size, bias=True)

        self.gate_fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=True)
        self.query_attentive_pooling = AttentivePooling(input_size=config.hidden_size)

        self.bigru1 = nn.GRU(config.hidden_size, config.hidden_size, 2, bidirectional=True)
        self.bigru2 = nn.GRU(config.hidden_size, config.hidden_size, 2, bidirectional=True)

        self.w_selfattn = nn.Linear(config.hidden_size, 1, bias=True)
        self.w_output = nn.Linear(config.hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.proj_attention = SCAttention(config.hidden_size, config.hidden_size)
        self.cnn_proj_attention = SCAttention(config.hidden_size, config.hidden_size)
        self.pooler = RobertaPooler(config)
        self.cnn_pooler = RobertaPooler(config)

        self.gate_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        self.filter_size = 3
        self.cnn = CNN_conv1d(config, filter_size=self.filter_size)

        self.svo_weight = svo_weight
        self.binary_weight = 0.5
        self.init_weights()

    def __init__hidden(self, batch_size):
        h0 = torch.rand(batch_size*self.num_labels, 1, self.hidden_size) # (2, batch_size, hidden_size)
        return h0.cuda()
    
    # def option_attention(self, U, V):
    #     S = torch.zeros(U.size(0), V.size(0))
    #     for i in range(U.size(0)):
    #         for j in range(V.size(0)):
    #             cat_result = torch.cat((U[i], V[j]))
    #             cat_result = torch.cat((cat_result, U[i]*V[j])).unsqueeze(1)    
    #             S[i][j] = torch.matmul(self.v,cat_result)
    #     A = torch.softmax(S, dim=0)
    #     assert A.shape == S.shape
    #     return A.to(self.device)             # [U.shape[0], V.shape[0]]

    # def gather_option(self, subset):
    #     subset_example_option = []
    #     for idx_i, i in enumerate(subset):
    #         cur_option = i
    #         for idx_j, j in enumerate(subset):
    #             if idx_j != idx_i:
    #                 attention_result = torch.matmul(j.T, self.option_attention(j, i)).T
    #                 attention_result = torch.cat((i-attention_result, i * attention_result), dim=1)
    #                 cur_option = torch.cat((cur_option, attention_result), dim=1)
    #             else:
    #                 continue
    #         subset_example_option.append(torch.tanh(self.W_c(cur_option)))
    #     return subset_example_option


    # def correlated_option(self, option_rep, num_choice=4):
    #     correlated_option = []
    #     for i in range(0, len(option_rep), num_choice):
    #         subset_per_example = option_rep[i: i + num_choice]
    #         subset_example_option = self.gather_option(subset_per_example)
    #         correlated_option.append(item for item in subset_example_option)
    #     return correlated_option

    def split_bert_sequence(self, seq, seq1_lengths, max_seq1_length, seq2_lengths, max_seq2_length, pad = 0, has_cls = True):
        if has_cls:
            cls = seq[:, 0, :]
        else:
            cls = None
        begin_index = 1 if has_cls else 0
        seq1 = seq[:, begin_index: max_seq1_length + begin_index, :]
        seq1_mask = self.sequence_mask(seq1_lengths, max_seq1_length)
        seq1 = seq1.float().masked_fill(
            ~seq1_mask.unsqueeze(2),
            float(pad),
        ).type_as(seq1)
        seq_range = torch.arange(0, max_seq2_length).long().unsqueeze(0)
        if seq1_lengths.is_cuda:
            seq_range = seq_range.cuda()
        seq_index = seq_range + seq1_lengths.unsqueeze(1) + 1 + begin_index
        batch_size, index_len = seq_index.size()
        dim = seq.size()[2]
        seq2 = torch.gather(seq, dim=1, index=seq_index.unsqueeze(2).expand(batch_size, index_len, dim))
        seq2_mask = self.sequence_mask(seq2_lengths, max_seq2_length)
        seq2 = seq2.float().masked_fill(
            ~seq2_mask.unsqueeze(2),
            float(pad),
        ).type_as(seq2)
        return cls, seq1, seq2

    def sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand

    def logic_op(self, input, input_mask):
        selfattn_unmask = self.w_selfattn(self.dropout(input))
        selfattn_unmask.masked_fill_(~input_mask, -float('inf'))
        selfattn_weight = F.softmax(selfattn_unmask, dim=1)
        selfattn = torch.sum(selfattn_weight * input, dim=1)
        score = self.w_output(self.dropout(selfattn))
        return score


    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # def forward(
    #     self,
    #     input_ids=None,        #[batch_size, num_choice, max_seq_len]
    #     token_type_ids=None,
    #     attention_mask=None,
    #     labels=None,
    #     position_ids=None,
    #     head_mask=None,
    #     inputs_embeds=None,
    #     output_attentions=None,
    #     output_hidden_states=None,
    #     return_dict=None,
    #     query_len=None,
    #     opt_len=None,
    #     svo_ids=None,
    #     # masked_ids=None,
    #     # masked_attention=None,
    # ):
    #     r"""
    #     labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
    #         Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
    #         num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
    #         :obj:`input_ids` above)
    #     """
    #     max_svo_len = svo_ids.size(2)
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #     num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
    #     bsz = input_ids.size(0)
    #     svo_graph_ids = svo_ids.view(bsz * self.num_labels, max_svo_len, 3, 2)
    #     flat_svo_ids = svo_ids.view(bsz * self.num_labels, -1, 2)
    #     flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None    
    #     flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
    #     flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
    #     # flat_choice_ids = choice_ids.view(-1, choice_ids.size(-1)) if choice_ids is not None else None
    #     flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
    #     flat_inputs_embeds = (
    #         inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
    #         if inputs_embeds is not None
    #         else None
    #     )
    #     # after flat: [bz*num_choice, max_seq_length]

    #     outputs = self.roberta(
    #         flat_input_ids,
    #         position_ids=flat_position_ids,
    #         token_type_ids=flat_token_type_ids,
    #         attention_mask=flat_attention_mask,
    #         head_mask=head_mask,
    #         inputs_embeds=flat_inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #     # masked_option = self.roberta(masked_ids, attention_mask=masked_attention)[0]
    #     # masked_option_contrastive = masked_option[:, 0, :].view(bsz, 1, self.hidden_size)

    #     sequence_output = outputs[0] # outputs[0].shape: [bz*num_choice, max_seq_length, dim]
    #     pooled_output = outputs[1]

    #     sequence_len = sequence_output.size(1)


    #     ### SVO triplet implementation
    #     batch_id = 0
    #     batch_svo_ids = []

    #     for batch in flat_svo_ids:
    #         offset = batch_id * sequence_len + 1
    #         svo_seqs = [offset + int(item[0]) for item in batch if int(item[0]) != -100]
    #         while (len(svo_seqs) < max_svo_len * 3):
    #             svo_seqs.append(0)
    #         batch_svo_ids.append(svo_seqs)
    #         batch_id += 1

    #     batch_svo_ids = torch.tensor(batch_svo_ids)
    #     batch_svo_ids = batch_svo_ids.view(-1)


    #     ### averaged svo_loss
    #     # for batch in flat_svo_ids:
    #     #     offset = batch_id * sequence_len + 1
    #     #     svo_seqs = [[offset + item[0], item[1]] for item in batch if item[0] != -100]
    #     #     while (len(svo_seqs) < max_svo_len * 3):
    #     #         svo_seqs.append([0, 0])
    #     #     batch_svo_ids.append(svo_seqs)
    #     #     batch_id += 1
    #     # batch_svo_ids = torch.tensor(batch_svo_ids)  # [bz*num_choice*max_svo_len *3]
    #     # batch_svo_ids = batch_svo_ids.view(-1, 2)

    #     sequence_output_svo = sequence_output.view(-1, self.hidden_size)
    #     sequence_output_svo = torch.cat([sequence_output_svo.new_zeros((1, self.hidden_size)), sequence_output_svo], dim=0)

    #     batch_svo_ids = batch_svo_ids.cuda()
    #     # svo_features = []
    #     # for item in batch_svo_ids:
    #     #     index = torch.tensor([i for i in range(item[0], item[0] + item[1])], dtype=torch.long).to(self.device)
    #     #     value = sequence_output_svo.index_select(0, index)
    #     #     if value.size(0) != 0 and value.size(0) != 1:
    #     #         value = (torch.sum(value, dim=0)/int(item[1])).unsqueeze(0) 
    #     #     svo_features.append(value)
    #     # svo_features = torch.cat(svo_features,dim=0).cuda()
    #     svo_features = sequence_output_svo.index_select(0, batch_svo_ids)
    #     svo_features = svo_features.view(bsz * self.num_labels, max_svo_len, 3, self.hidden_size)
    #     # svo_features = svo_features.view(-1, 3, self.hidden_size)
    #     # eou_features = eou_features.cuda()

    #     sv_features = svo_features[:, :, 0, :] + svo_features[:, :, 1, :]
    #     o_features = svo_features[:, :, 2, :]
    #     # sv_features = svo_features[:, 0, :] + svo_features[:, 1, :]
    #     # o_features = svo_features[:, 2, :]

    #     ### SVO graph implementation
    #     svo_rep = []
    #     svo_mask = []
    #     for choice_idx, svo in enumerate(svo_graph_ids):
    #         G = dgl.DGLGraph().to(self.device)
    #         edge_type = []
    #         edges = []
    #         nodes_all = []
    #         for item in svo.view(-1,  2):
    #             if list(item) not in nodes_all:
    #                 nodes_all.append(list(item))
    #         if [-100, 0] in nodes_all:
    #             nodes_all.remove([-100,0])
    #         G.add_nodes(len(nodes_all))

    #         if nodes_all == []:
    #             svo = torch.tensor([[[0, 1], [self.max_doc_len +2, 1], [self.max_doc_len + self.max_query_len + self.max_option_len +4, 1]]]).to(self.device)
    #             nodes_all = [list(item) for item in svo.view(-1, 2)]
            

    #         for item in svo:
    #             if int(item[0][0]) == -100:
    #                 break
    #             else:
    #                 # add default_in and default_out edges
    #                 G.add_edges(nodes_all.index(list(item[0])), nodes_all.index(list(item[1])))
    #                 edges.append([nodes_all.index(list(item[0])), nodes_all.index(list(item[1]))])
    #                 edge_type.append(0)

    #                 G.add_edges(nodes_all.index(list(item[1])), nodes_all.index(list(item[0])))
    #                 edges.append([nodes_all.index(list(item[1])), nodes_all.index(list(item[0]))])
    #                 edge_type.append(1)

    #                 G.add_edges(nodes_all.index(list(item[1])), nodes_all.index(list(item[2])))
    #                 edges.append([nodes_all.index(list(item[1])), nodes_all.index(list(item[2]))])
    #                 edge_type.append(2)

    #                 G.add_edges(nodes_all.index(list(item[2])), nodes_all.index(list(item[1])))
    #                 edges.append([nodes_all.index(list(item[2])), nodes_all.index(list(item[1]))])
    #                 edge_type.append(3)
    #         # add self edges
    #         for item in range(len(nodes_all)):
    #             G.add_edge(item, item)
    #             edge_type.append(4)
    #             edges.append([item, item])

    #         # add node feature
    #         for i in range(len(nodes_all)):
    #             # index = torch.tensor([i for i in range(nodes_all[i][0], nodes_all[i][0] + nodes_all[i][1])], dtype=torch.long).to(self.device)
    #             # selected_item = torch.index_select(sequence_output[choice_idx], 0, index)
    #             # if selected_item.size(0) == 1:
    #             #     value = selected_item
    #             # else:
    #             #     value = (torch.sum(selected_item, dim=0)/nodes_all[i][1]).unsqueeze(0)
    #             # G.nodes[[i]].data['h'] = value
    #             index = torch.tensor([nodes_all[i][0],], dtype=torch.long).to(self.device)
    #             selected_item = torch.index_select(sequence_output[choice_idx], 0, index)
    #             G.nodes[[i]].data['h'] = selected_item
            
    #         edge_norm = []
    #         for e1, e2 in edges:
    #             if e1 == e2:
    #                 edge_norm.append(1)
    #             else:
    #                 edge_norm.append(1/(G.in_degrees(e2)-1))
            
    #         edge_type = torch.from_numpy(np.array(edge_type)).to(self.device)
    #         edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().to(self.device)

    #         G.edata.update({'rel_type':edge_type,})
    #         G.edata.update({'norm': edge_norm})
    #         X = self.GCN(G)[0]
    #         svo_rep.append(X)
    #         svo_mask.append(torch.tensor([1] * X.size(0), dtype=torch.bool))
    #     svo_rep = torch.nn.utils.rnn.pad_sequence(svo_rep).to(self.device)
    #     svo_rep = torch.transpose(svo_rep, 0, 1).contiguous()
    #     svo_mask = torch.nn.utils.rnn.pad_sequence(svo_mask).to(self.device).unsqueeze(2)
    #     svo_mask = torch.transpose(svo_mask, 0, 1).contiguous()
    #     # final_rep = self.logic_op(svo_rep, svo_mask)
    #     output = self.proj_attention(sequence_output, flat_attention_mask.unsqueeze(2), svo_rep, svo_mask)
    #     merge = torch.cat((sequence_output, output), dim=-1)
    #     gate = self.sigmoid(self.gate_dense(merge))
    #     output = (1-gate) * sequence_output + gate * output
    #     final_rep = self.pooler(output)

    #     ## CNN
    #     if svo_rep.size(1) % self.filter_size == 0:
    #         cnn_output = self.cnn(svo_rep, self.filter_size)
    #         cnn_output_mask = torch.ones(svo_rep.size(0), int(svo_rep.size(1)/3), 1).to(self.device)
    #     else:
    #         cnn_mask_padded = torch.zeros(svo_rep.size(0), 1, 1)
    #         cnn_mask = torch.ones(svo_rep.size(0), int(svo_rep.size(1)/3), 1)
    #         zeros = torch.zeros(svo_rep.size(0), 3-svo_rep.size(1)%3, svo_rep.size(2)).cuda()
    #         padded = torch.cat((svo_rep, zeros), dim=1)
    #         cnn_output = self.cnn(padded, self.filter_size)
    #         cnn_output_mask = torch.cat((cnn_mask, cnn_mask_padded), dim=1).to(self.device) 

    #     cnn_output = self.cnn_proj_attention(sequence_output, flat_attention_mask.unsqueeze(2), cnn_output, cnn_output_mask)
    #     cnn_output = self.pooler(cnn_output)

    #     # assert sequence_output.size(0) == flat_choice_ids.size(0)
    #     # assert sequence_output.size(1) == flat_choice_ids.size(1)

    #     # doc_enc = sequence_output[:, 0 : self.max_doc_len + 2, :]
    #     # opt_enc = sequence_output[:, self.max_doc_len + 2:, :]          # opt_enc.shape[1] == masked_option.shape[1]

    #     # opt_len = opt_len.view(-1)
    #     # query_len = query_len.view(-1)

    #     # _, query_enc, _ = self.split_bert_sequence(opt_enc, query_len, self.max_query_len, opt_len, self.max_option_len, has_cls=True)

    #     # options_rep = []
    #     # for idx, item in enumerate(sequence_output):
    #     #     select_index =[]
    #     #     for id, items in enumerate(flat_choice_ids[idx]):
    #     #         if items == 1:
    #     #             select_index.append(id)
    #     #         else:
    #     #             continue
    #     #     select_index = torch.LongTensor(select_index).cuda()
    #     #     selected_option = torch.index_select(item, 0, select_index)
    #     #     options_rep.append(selected_option)
    #     # correlated_option_rep = self.correlated_option(options_rep, num_choices)

    #     # query_total_len = query_enc.size(1)
    #     # query_mask = self.sequence_mask(query_len, query_total_len)
    #     # query_attn = self.query_attentive_pooling(query_enc, query_mask)

    #     # opt_total_len = opt_enc.size(1)
    #     # opt_mask = flat_attention_mask[:, self.max_doc_len + 2:] > 0
    #     # doc_mask = flat_attention_mask[:, 0 : self.max_doc_len + 2] > 0
        
    #     # opt_mask = opt_mask.view(bsz, self.num_labels, opt_total_len)
    #     # opt_enc = opt_enc.view(bsz, self.num_labels, opt_total_len, self.hidden_size)

    #     # compute contrative scores
    #     # h0 = self.__init__hidden(bsz)
    #     # h1 = self.__init__hidden(bsz)
    #     # self.bigru2.flatten_parameters
    #     # self.bigru1.flatten_parameters()
    #     # out1, _ = self.bigru1(opt_enc[:, :, 0, :].view(bsz*self.num_labels, 1, self.hidden_size), h0)
    #     # out2, _ = self.bigru2(masked_option_contrastive, h1)

    #     # opt_enc_contrastive = out1.view(bsz, self.num_labels, 2 * self.hidden_size)
    #     # contrastive_scores = torch.bmm(opt_enc_contrastive, torch.transpose(out2, 1, 2))
    #     # contrastive_scores = F.softmax(contrastive_scores).view(bsz, self.num_labels)

    #     ### Option Comparison
    #     # correlation_list = []
    #     # for i in range(self.num_labels):
    #     #     cur_opt = opt_enc[:, i, :, :]
    #     #     cur_mask = opt_mask[:, i, :]

    #     #     comp_info = []
    #     #     for j in range(self.num_labels):
    #     #         if j == i:
    #     #             continue

    #     #         tmp_opt = opt_enc[:, j, :, :]
    #     #         tmp_mask = opt_mask[:, j, :]

    #     #         (attn, _), _ = self.opt_attention(cur_opt, tmp_opt, tmp_opt, cur_mask, tmp_mask)
    #     #         comp_info.append(cur_opt * attn)
    #     #         comp_info.append(cur_opt - attn)

    #     #     correlation = torch.tanh(self.comp_fc(torch.cat([cur_opt] + comp_info, dim=-1)))
    #     #     correlation_list.append(correlation)

    #     # correlation_list = [correlation.unsqueeze(1) for correlation in correlation_list]
    #     # opt_correlation = torch.cat(correlation_list, dim=1)

    #     # opt_mask = opt_mask.view(bsz * self.num_labels, opt_total_len)
    #     # opt_enc = opt_enc.view(bsz * self.num_labels, opt_total_len, self.hidden_size)
    #     # opt_correlation = opt_correlation.contiguous().view(bsz * self.num_labels, opt_total_len, self.hidden_size)
    #     # gate = torch.sigmoid(self.gate_fc(torch.cat((opt_enc, opt_correlation, query_attn.expand_as(opt_enc)), -1)))
    #     # option = opt_enc * gate + opt_correlation * (1.0 - gate)

    #     # (attn, _), (coattn, _) = self.attention(option, doc_enc, doc_enc, opt_mask, doc_mask)
    #     # fusion = self.attn_fc(torch.cat((option, attn, coattn), -1))
    #     # fusion = F.relu(fusion)

    #     # (attn, _), _ = self.opt_self_attention(fusion, fusion, fusion, opt_mask, opt_mask)
    #     # fusion = self.opt_selfattn_fc(torch.cat((fusion, attn, fusion * attn, fusion - attn), -1))
    #     # fusion = F.relu(fusion)

    #     # fusion = fusion.masked_fill(
    #     #     ~opt_mask.unsqueeze(-1).expand(bsz * self.num_labels, opt_total_len, self.hidden_size),
    #     #     -float('inf'))
    #     # fusion, _ = fusion.max(dim=1)

    #     # fusion = self.dropout2(fusion)
    #     # reshaped_logits = self.score_fc(fusion).view(bsz, self.num_labels)

    #     # pooled_output = self.dropout1(pooled_output)
    #     # logits = self.classifier(pooled_output)
    #     # reshaped_logits = (logits.view(-1, num_choices) + reshaped_logits)/2
    #     # reshaped_logits = logits.view(-1, num_choices)
    #     # svo_logits = final_rep.view(-1, num_choices)
    #     # reshaped_logits = F.softmax((svo_logits.view(-1, num_choices) + reshaped_logits)/2)
    #     logits = self.classifier(final_rep)
    #     reshaped_logits = logits.view(-1, num_choices)

    #     loss = None
    #     if labels is not None:
    #         # binary_labels = torch.zeros_like(reshaped_logits)
    #         # for bid, choice in enumerate(labels):
    #         #     binary_labels[bid][choice] = 1
            
    #         loss_fct = CrossEntropyLoss()
    #         # loss_binary = nn.BCEWithLogitsLoss()
    #         # loss_contrastive = loss_fct(contrastive_scores, labels)
    #         cls_loss = loss_fct(reshaped_logits, labels)
    #         criterion = nn.CosineSimilarity()
    #         svo_sim = criterion(sv_features.view(-1, self.hidden_size), o_features.view(-1, self.hidden_size))
    #         svo_inds = torch.nonzero(svo_sim)
    #         svo_sim = svo_sim[svo_inds]
    #         svo_loss = 1. - svo_sim
    #         svo_loss = torch.mean(svo_loss)
    #         # svo_graph_loss = loss_fct(svo_logits, labels)
    #         total_loss = cls_loss + self.svo_weight * svo_loss
            
    #     if not return_dict:
    #         output = (reshaped_logits,) + outputs[2:]
    #         return ((total_loss,) + output) if total_loss is not None else output

    #     return MultipleChoiceModelOutput(
    #         loss=total_loss,
    #         logits=reshaped_logits,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )
    def forward(
        self,
        input_ids=None,        #[batch_size, num_choice, max_seq_len]
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # choice_ids=None,
        query_len=None,
        opt_len=None,
        svo_ids=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        max_svo_len = svo_ids.size(2)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        bsz = input_ids.size(0)
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None    
        svo_graph_ids = svo_ids.view(bsz * self.num_labels, max_svo_len, 3, 2)
        flat_svo_ids = svo_ids.view(bsz * self.num_labels, -1, 2)
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # flat_choice_ids = choice_ids.view(-1, choice_ids.size(-1)) if choice_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        # after flat: [bz*num_choice, max_seq_length]

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] # outputs[0].shape: [bz*num_choice, max_seq_length, dim]
        pooled_output = outputs[1]
        # sequence_output = pooled_output
        # assert sequence_output.size(0) == flat_choice_ids.size(0)
        # assert sequence_output.size(1) == flat_choice_ids.size(1)
        sequence_len = sequence_output.size(1)


        # ### mask contrastive implementation
        # masked_attention = attention_mask.clone()
        # masked_input_ids = input_ids.clone()
        # attention = []
        # input_ids_ = []
        # for i, item in enumerate(masked_input_ids):
        #     r1 = random.randint(0, input_ids.size(2)-1)
        #     r2 = random.randint(0, input_ids.size(2)-1)
        #     r3 = random.randint(0, input_ids.size(2)-1)
        #     item[int(labels[i])][r1] = 50264
        #     item[int(labels[i])][r2] = 50264
        #     item[int(labels[i])][r3] = 50264
        #     masked_attention[i][int(labels[i])][r1] = 0
        #     masked_attention[i][int(labels[i])][r2] = 0
        #     masked_attention[i][int(labels[i])][r3] = 0
        #     attention.append(masked_attention[i][int(labels[i])])
        #     input_ids_.append(item[int(labels[i])])
        # attention = torch.cat(attention).view(bsz, 1, sequence_len)
        # input_ids_ = torch.cat(input_ids_).view(bsz, 1, sequence_len)
        
        # masked_attention = attention.view(-1, attention.size(-1))
        # masked_input_ids = input_ids_.view(-1, input_ids_.size(-1))
        # masked_outputs = self.roberta(masked_input_ids, attention_mask=masked_attention)[0]

        # ###  compute contrative scores
        # h0 = self.__init__hidden(bsz)
        # h1 = self.__init__hidden(bsz)
        # self.bigru2.flatten_parameters()
        # self.bigru1.flatten_parameters()
        # out1, _ = self.bigru1(sequence_output[:, 0, :].view(sequence_output.size(0), 1, sequence_output.size(2)), h0)
        # out2, _ = self.bigru2(masked_outputs[:, 0, :].view(masked_outputs.size(0), 1, masked_outputs.size(2)), h1)

        # out1_contrastive = out1
        # out2_contrastive = out2
        # opt_enc_contrastive = out1_contrastive.view(bsz, self.num_labels, 2 * self.hidden_size)
        # contrastive_scores = torch.bmm(opt_enc_contrastive, torch.transpose(out2_contrastive, 1, 2))
        # contrastive_scores = F.softmax(contrastive_scores).view(bsz, self.num_labels)

        # ### SVO triplet implementation
        # batch_id = 0
        # batch_svo_ids = []

        # for batch in flat_svo_ids:
        #     offset = batch_id * sequence_len + 1
        #     svo_seqs = [offset + int(item[0]) for item in batch if int(item[0]) != -100]
        #     while (len(svo_seqs) < max_svo_len * 3):
        #         svo_seqs.append(0)
        #     batch_svo_ids.append(svo_seqs)
        #     batch_id += 1

        # batch_svo_ids = torch.tensor(batch_svo_ids)
        # batch_svo_ids = batch_svo_ids.view(-1)


        # ### averaged svo_loss
        # # for batch in flat_svo_ids:
        # #     offset = batch_id * sequence_len + 1
        # #     svo_seqs = [[offset + item[0], item[1]] for item in batch if item[0] != -100]
        # #     while (len(svo_seqs) < max_svo_len * 3):
        # #         svo_seqs.append([0, 0])
        # #     batch_svo_ids.append(svo_seqs)
        # #     batch_id += 1
        # # batch_svo_ids = torch.tensor(batch_svo_ids)  # [bz*num_choice*max_svo_len *3]
        # # batch_svo_ids = batch_svo_ids.view(-1, 2)

        # sequence_output_svo = sequence_output.view(-1, self.hidden_size)
        # sequence_output_svo = torch.cat([sequence_output_svo.new_zeros((1, self.hidden_size)), sequence_output_svo], dim=0)

        # batch_svo_ids = batch_svo_ids.to(self.device)
        # # svo_features = []
        # # for item in batch_svo_ids:
        # #     index = torch.tensor([i for i in range(item[0], item[0] + item[1])], dtype=torch.long).to(self.device)
        # #     value = sequence_output_svo.index_select(0, index)
        # #     if value.size(0) == 0:
        # #         print(value.shape)
        # #         value = torch.tensor([value])
        # #     if value.size(0) != 0 and value.size(0) != 1:
        # #         value = torch.sum(value, dim=0)/int(item[1]) 
        # #     svo_features.append(value)
        # # svo_features = torch.cat(svo_features,dim=0).cuda()
        # svo_features = sequence_output_svo.index_select(0, batch_svo_ids)
        # svo_features = svo_features.view(bsz * self.num_labels, max_svo_len, 3, self.hidden_size)
        # # eou_features = eou_features.cuda()

        # sv_features = svo_features[:, :, 0, :] + svo_features[:, :, 1, :]
        # o_features = svo_features[:, :, 2, :]

        # ### SVO graph implementation
        # svo_rep = []
        # svo_mask = []
        # for choice_idx, svo in enumerate(svo_graph_ids):
        #     G = dgl.DGLGraph().to(self.device)
        #     edge_type = []
        #     edges = []
        #     nodes_all = []
        #     for item in svo.view(-1,  2):
        #         if list(item) not in nodes_all:
        #             nodes_all.append(list(item))
        #     if [-100, 0] in nodes_all:
        #         nodes_all.remove([-100,0])
        #     G.add_nodes(len(nodes_all))

        #     if nodes_all == []:
        #         svo = torch.tensor([[[0, 1], [self.max_doc_len +2, 1], [self.max_doc_len + self.max_query_len + self.max_option_len +4, 1]]]).to(self.device)
        #         nodes_all = [list(item) for item in svo.view(-1, 2)]
            

        #     for item in svo:
        #         if int(item[0][0]) == -100:
        #             break
        #         else:
        #             # add default_in and default_out edges
        #             G.add_edges(nodes_all.index(list(item[0])), nodes_all.index(list(item[1])))
        #             edges.append([nodes_all.index(list(item[0])), nodes_all.index(list(item[1]))])
        #             edge_type.append(0)

        #             G.add_edges(nodes_all.index(list(item[1])), nodes_all.index(list(item[0])))
        #             edges.append([nodes_all.index(list(item[1])), nodes_all.index(list(item[0]))])
        #             edge_type.append(1)

        #             G.add_edges(nodes_all.index(list(item[1])), nodes_all.index(list(item[2])))
        #             edges.append([nodes_all.index(list(item[1])), nodes_all.index(list(item[2]))])
        #             edge_type.append(2)

        #             G.add_edges(nodes_all.index(list(item[2])), nodes_all.index(list(item[1])))
        #             edges.append([nodes_all.index(list(item[2])), nodes_all.index(list(item[1]))])
        #             edge_type.append(3)
        #     # add self edges
        #     for item in range(len(nodes_all)):
        #         G.add_edge(item, item)
        #         edge_type.append(4)
        #         edges.append([item, item])

        #     # add node feature
        #     for i in range(len(nodes_all)):
        #         # index = torch.tensor([nodes_all[i][0], ], dtype=torch.long).to(self.device)
        #         index = torch.tensor([i for i in range(nodes_all[i][0], nodes_all[i][0] + nodes_all[i][1])], dtype=torch.long).to(self.device)
        #         selected_item = torch.index_select(sequence_output[choice_idx], 0, index)
        #         if selected_item.size(0) == 1:
        #             value = selected_item
        #         else:
        #             value = (torch.sum(selected_item, dim=0)/nodes_all[i][1]).unsqueeze(0)
        #         G.nodes[[i]].data['h'] = value
            
        #     edge_norm = []
        #     for e1, e2 in edges:
        #         if e1 == e2:
        #             edge_norm.append(1)
        #         else:
        #             edge_norm.append(1/(G.in_degrees(e2)-1))
            
        #     edge_type = torch.from_numpy(np.array(edge_type)).to(self.device)
        #     edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().to(self.device)

        #     G.edata.update({'rel_type':edge_type,})
        #     G.edata.update({'norm': edge_norm})
        #     X = self.GCN(G)[0]
        #     svo_rep.append(X)
        #     svo_mask.append(torch.tensor([1] * X.size(0), dtype=torch.bool))
        # svo_rep = torch.nn.utils.rnn.pad_sequence(svo_rep).to(self.device)
        # svo_rep = torch.transpose(svo_rep, 0, 1).contiguous()
        # svo_mask = torch.nn.utils.rnn.pad_sequence(svo_mask).to(self.device).unsqueeze(2)
        # svo_mask = torch.transpose(svo_mask, 0, 1).contiguous()
        # # final_rep = self.logic_op(svo_rep, svo_mask)
        # output = self.proj_attention(sequence_output, flat_attention_mask.unsqueeze(2), svo_rep, svo_mask)
        # merge = torch.cat((sequence_output, output), dim=-1)
        # gate = self.sigmoid(self.gate_dense(merge))
        # output = (1-gate) * sequence_output + gate * output
        # final_rep = self.pooler(output)

        # ### OCN implementation
        # doc_enc = sequence_output[:, 0 : self.max_doc_len + 2, :]
        # opt_enc = sequence_output[:, self.max_doc_len + 2:, :]

        # opt_len = opt_len.view(-1)
        # query_len = query_len.view(-1)

        # _, query_enc, _ = self.split_bert_sequence(opt_enc, query_len, self.max_query_len, opt_len, self.max_option_len, has_cls=True)

        # # options_rep = []
        # # for idx, item in enumerate(sequence_output):
        # #     select_index =[]
        # #     for id, items in enumerate(flat_choice_ids[idx]):
        # #         if items == 1:
        # #             select_index.append(id)
        # #         else:
        # #             continue
        # #     select_index = torch.LongTensor(select_index).cuda()
        # #     selected_option = torch.index_select(item, 0, select_index)
        # #     options_rep.append(selected_option)
        # # correlated_option_rep = self.correlated_option(options_rep, num_choices)

        # query_total_len = query_enc.size(1)
        # query_mask = self.sequence_mask(query_len, query_total_len)
        # query_attn = self.query_attentive_pooling(query_enc, query_mask)

        # opt_total_len = opt_enc.size(1)
        # opt_mask = flat_attention_mask[:, self.max_doc_len + 2:] > 0
        # doc_mask = flat_attention_mask[:, 0 : self.max_doc_len + 2] > 0
        
        # opt_mask = opt_mask.view(bsz, self.num_labels, opt_total_len)
        # opt_enc = opt_enc.view(bsz, self.num_labels, opt_total_len, self.hidden_size)

        # ### Option Comparison
        # correlation_list = []
        # for i in range(self.num_labels):
        #     cur_opt = opt_enc[:, i, :, :]
        #     cur_mask = opt_mask[:, i, :]

        #     comp_info = []
        #     for j in range(self.num_labels):
        #         if j == i:
        #             continue

        #         tmp_opt = opt_enc[:, j, :, :]
        #         tmp_mask = opt_mask[:, j, :]

        #         (attn, _), _ = self.opt_attention(cur_opt, tmp_opt, tmp_opt, cur_mask, tmp_mask)
        #         comp_info.append(cur_opt * attn)
        #         comp_info.append(cur_opt - attn)

        #     correlation = torch.tanh(self.comp_fc(torch.cat([cur_opt] + comp_info, dim=-1)))
        #     correlation_list.append(correlation)

        # correlation_list = [correlation.unsqueeze(1) for correlation in correlation_list]
        # opt_correlation = torch.cat(correlation_list, dim=1)

        # opt_mask = opt_mask.view(bsz * self.num_labels, opt_total_len)
        # opt_enc = opt_enc.view(bsz * self.num_labels, opt_total_len, self.hidden_size)
        # opt_correlation = opt_correlation.contiguous().view(bsz * self.num_labels, opt_total_len, self.hidden_size)
        # gate = torch.sigmoid(self.gate_fc(torch.cat((opt_enc, opt_correlation, query_attn.expand_as(opt_enc)), -1)))
        # option = opt_enc * gate + opt_correlation * (1.0 - gate)

        # (attn, _), (coattn, _) = self.attention(option, doc_enc, doc_enc, opt_mask, doc_mask)
        # fusion = self.attn_fc(torch.cat((option, attn, coattn), -1))
        # fusion = F.relu(fusion)

        # (attn, _), _ = self.opt_self_attention(fusion, fusion, fusion, opt_mask, opt_mask)
        # fusion = self.opt_selfattn_fc(torch.cat((fusion, attn, fusion * attn, fusion - attn), -1))
        # fusion = F.relu(fusion)

        # fusion = fusion.masked_fill(
        #     ~opt_mask.unsqueeze(-1).expand(bsz * self.num_labels, opt_total_len, self.hidden_size),
        #     -float('inf'))
        # fusion, _ = fusion.max(dim=1)

        # fusion = self.dropout2(fusion)
        # reshaped_logits = self.score_fc(fusion).view(bsz, self.num_labels)

        # pooled_output = self.dropout1(pooled_output)
        # logits = self.classifier(pooled_output)
        # reshaped_logits = logits.view(-1, num_choices)
        # svo_logits = final_rep.view(-1, num_choices)
        # reshaped_logits = (svo_logits.view(-1, num_choices) + reshaped_logits)/2
        # logits = self.classifier(final_rep)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # binary_labels = torch.zeros_like(reshaped_logits)
            # for bid, choice in enumerate(labels):
            #     binary_labels[bid][choice] = 1
            # loss_binary = nn.BCEWithLogitsLoss()
            # binary_loss = loss_binary(reshaped_logits, binary_labels)

            loss_fct = CrossEntropyLoss()
            cls_loss = loss_fct(reshaped_logits, labels)
            # criterion = nn.CosineSimilarity()
            # svo_sim = criterion(sv_features.view(-1, self.hidden_size), o_features.view(-1, self.hidden_size))
            # svo_inds = torch.nonzero(svo_sim)
            # svo_sim = svo_sim[svo_inds]
            # svo_loss = 1. - svo_sim
            # svo_loss = torch.mean(svo_loss)

            # loss_contrastive = loss_fct(contrastive_scores, labels)
            total_loss = cls_loss # + self.svo_weight * svo_loss #  + 0.3 * loss_contrastive     #  + self.binary_weight * binary_loss
            # print(svo_loss)
            
            # loss = loss_fct(reshaped_logits, labels) + loss_binary(reshaped_logits, binary_labels)
            

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MultipleChoiceModelOutput(
            loss=total_loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

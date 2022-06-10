import torch
import torch.nn.functional as nn
import transformers.modeling_bert as mb
import os
import collections
from ditto.exceptions import ModelNotFoundError


class MyBERTModel(mb.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = mb.BertEncoder(config)
        self.pooler = mb.BertPooler(config)

        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

def SplitModel(path):
    path = os.path.join(path, 'pytorch_model.bin')
    if not os.path.exists(path):
        raise ModelNotFoundError(path)
    saved_state = torch.load(path, map_location=lambda storage, loc: storage)
    BertEmbeds = collections.OrderedDict()
    BertEmbeds['bert.embeddings.word_embeddings.weight'] = saved_state.pop('bert.embeddings.word_embeddings.weight')
    BertEmbeds['bert.embeddings.position_embeddings.weight'] = saved_state.pop('bert.embeddings.position_embeddings.weight')
    BertEmbeds['bert.embeddings.token_type_embeddings.weight'] = saved_state.pop('bert.embeddings.token_type_embeddings.weight')
    BertEmbeds['bert.embeddings.LayerNorm.weight'] = saved_state.pop('bert.embeddings.LayerNorm.weight')
    BertEmbeds['bert.embeddings.LayerNorm.bias'] = saved_state.pop('bert.embeddings.LayerNorm.bias')
    return BertEmbeds, saved_state


def _BertEmbedding(input_ids,BertEmbeds,position_ids = None, token_type_ids = None, inputs_embeds = None,hidden_size = 512):
    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]
    hidden_size = (hidden_size,)

    seq_length = input_shape[1]
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    if inputs_embeds is None:
        inputs_embeds = nn.embedding(input_ids,BertEmbeds['bert.embeddings.word_embeddings.weight'])
    position_embeddings = nn.embedding(position_ids,BertEmbeds['bert.embeddings.position_embeddings.weight'])
    token_type_embeddings = nn.embedding(token_type_ids,BertEmbeds['bert.embeddings.token_type_embeddings.weight'])

    embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    embeddings = nn.layer_norm(embeddings,tuple(hidden_size),weight = BertEmbeds['bert.embeddings.LayerNorm.weight'], bias = BertEmbeds['bert.embeddings.LayerNorm.bias'],eps=1e-12)
    embeddings = nn.dropout(embeddings,p = 0.1,training=False)

    return embeddings
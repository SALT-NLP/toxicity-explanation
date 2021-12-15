from torch import nn, torch
from transformers import BartModel,BartForConditionalGeneration,BartPretrainedModel
from transformers.models.bart.modeling_bart import _expand_mask,shift_tokens_right,BartDecoder,BartEncoder,BartDecoderLayer,BartAttention,BartLearnedPositionalEmbedding
from transformers.activations import ACT2FN
from transformers.modeling_outputs import Seq2SeqLMOutput,BaseModelOutput,BaseModelOutputWithPastAndCrossAttentions
from transformers.file_utils import ModelOutput
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from inspect import signature
from tqdm import tqdm
from generation_utils import MultiViewGenerationMixin

import math
import random
import torch.nn.functional as F

class Seq2SeqMultiViewLMOutput(ModelOutput):
    loss = None
    logits = None
    past_key_values = None
    decoder_hidden_states = None
    decoder_attentions = None
    cross_attentions = None
    encoder_last_hidden_state = None
    encoder_hidden_states = None
    encoder_attentions = None
    view_attention = None

class Seq2SeqMultiViewModelOutput(ModelOutput):
    last_hidden_state = None
    past_key_values = None
    decoder_hidden_states = None
    decoder_attentions = None
    cross_attentions = None
    encoder_last_hidden_state = None
    encoder_hidden_states = None
    encoder_attentions = None
    view_attention = None

class BartMultiViewAttention(BartAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout = 0.0,
        is_decoder = False,
        bias = True,
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    
    def forward(
        self,
        hidden_states, # hidden states
        key_value_states_by_view, # encoder hidden states
        past_key_value_by_view,
        attention_mask, # encoder attention mask
        view_attention,
        layer_head_mask,
        output_attentions,
        num_views,
    ):
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        bsz, tgt_len, embed_dim = hidden_states.size()
        
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        
        key_states_by_view = []
        value_states_by_view = []
        for i,key_value_state in enumerate(key_value_states_by_view):
          if past_key_value_by_view is not None:
              key_states_by_view.append(past_key_value_by_view[0][i])
              value_states_by_view.append(past_key_value_by_view[1][i])
          else:
              key_states_by_view.append(self._shape(self.k_proj(key_value_state), -1, bsz))
              value_states_by_view.append(self._shape(self.v_proj(key_value_state), -1, bsz))
        
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value_by_view = (key_states_by_view, value_states_by_view)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        
        attn_weights_reshaped = []
        attn_output = []
        for i, key_value_tup in enumerate(zip(key_states_by_view, value_states_by_view)):
          key_states = key_value_tup[0]
          value_states = key_value_tup[1]

          key_states = key_states.view(*proj_shape)
          value_states = value_states.view(*proj_shape)

          src_len = key_states.size(1)
          attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

          assert attn_weights.size() == (
              bsz * self.num_heads,
              tgt_len,
              src_len,
          ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
          
          indices = list(range(i, attention_mask.shape[0], num_views))
          attention_mask_i = attention_mask[indices, :]
          if attention_mask_i is not None:
              assert attention_mask_i.size() == (
                  bsz,
                  1,
                  tgt_len,
                  src_len,
              ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask_i.size()}"
              attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask_i
              attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

          attn_weights = F.softmax(attn_weights, dim=-1)

          if layer_head_mask is not None:
              assert layer_head_mask.size() == (
                  self.num_heads,
              ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
              attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
              attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

          if output_attentions:
              # this operation is a bit akward, but it's required to
              # make sure that attn_weights keeps its gradient.
              # In order to do so, attn_weights have to reshaped
              # twice and have to be reused in the following
              attn_weights_reshaped.append(attn_weights.view(bsz, self.num_heads, tgt_len, src_len))
              attn_weights = attn_weights_reshaped[-1].view(bsz * self.num_heads, tgt_len, src_len)
          else:
              attn_weights_reshaped = None

          attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

          attn_output.append(torch.bmm(attn_probs, value_states))

          assert attn_output[-1].size() == (
              bsz * self.num_heads,
              tgt_len,
              self.head_dim,
          ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

          attn_output[-1] = (
              attn_output[-1].view(bsz, self.num_heads, tgt_len, self.head_dim)
              .transpose(1, 2)
              .reshape(bsz, tgt_len, embed_dim)
          )

          attn_output[-1] = self.out_proj(attn_output[-1])
        
        attn_output = torch.stack(attn_output, dim=0)
        attn_output = attn_output.mul(view_attention.T[:, :, None, None]).sum(dim=0)

        return attn_output, attn_weights_reshaped, past_key_value_by_view

class BartMultiViewDecoderLayer(BartDecoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartMultiViewAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        hidden_states,
        attention_mask = None,
        encoder_hidden_states_by_view = None,
        encoder_attention_mask = None,
        view_attention=None,
        layer_head_mask = None,
        encoder_layer_head_mask = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = True,
        num_views = 6
    ):
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states_by_view is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states_by_view=encoder_hidden_states_by_view,
                attention_mask=encoder_attention_mask,
                view_attention=view_attention,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value_by_view=cross_attn_past_key_value,
                output_attentions=output_attentions,
                num_views=num_views
            )
            
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartMultiViewDecoder(BartDecoder):
    def __init__(self, config, embed_tokens = None):
        super(BartPretrainedModel, self).__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        layers = [BartMultiViewDecoderLayer(config) for _ in range(config.decoder_layers)]
        self.layers = nn.ModuleList(layers)
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states_by_view=None,
        encoder_attention_mask=None,
        view_attention=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        num_views=6,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states_by_view is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states_by_view is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states_by_view,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states_by_view=encoder_hidden_states_by_view,
                    encoder_attention_mask=encoder_attention_mask,
                    view_attention=view_attention,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    num_views=num_views,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states_by_view is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BartMultiViewModel(BartModel):
    def __init__(self, config):
        super(BartPretrainedModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartMultiViewDecoder(config, self.shared)

        self.init_weights()
    

class BartForConditionalGenerationMultiViewModel(BartForConditionalGeneration, MultiViewGenerationMixin):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight",r"view_weight\.bias",r"view_weight\.weight"]

    def __init__(
        self,
        config,
        num_views=6,
        use_cuda=True,
        T=0.5,
        tokenizer=None,
    ):
        super(BartPretrainedModel, self).__init__(config)
        self.model = BartMultiViewModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.use_cuda = use_cuda
        self.tokenizer = tokenizer

        # View params
        self.view_lstm = nn.LSTM(input_size=config.d_model, hidden_size=config.d_model, batch_first=True)
        self.view_proj = nn.Linear(config.d_model, config.d_model)
        self.view_proj_layer_norm = nn.LayerNorm(config.d_model)
        self.view_context_vector = nn.Linear(config.d_model, 1, bias=False)
 
        self.view_activation = nn.Tanh()
        self.view_align = nn.Softmax(dim=1)
 
        self.T = T
        self.num_views = num_views
 
        self.init_weights()

    def encoder_view_forward(
        self,
        input_ids,
        view_token_idx,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Housekeeping
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        
        encoder_outputs_by_view = []
        embs_by_view = []

        for i in range(self.num_views):
          indices = list(range(i, batch_size, self.num_views))
 
          encoder_outputs_by_view.append(
              self.model.encoder(
                  input_ids[indices],
                  attention_mask=attention_mask[indices],
                  head_mask=head_mask,
                  inputs_embeds=inputs_embeds,
                  output_attentions=output_attentions,
                  output_hidden_states=output_hidden_states,
                  return_dict=return_dict,
              )
          )
          
          hidden_state = encoder_outputs_by_view[-1].last_hidden_state
          
          # Gather View Embeddings
          view_embs = torch.gather(
            hidden_state, 1,
            view_token_idx[indices,:,None].repeat(1, 1, hidden_state.shape[2])
          )
          
          view_embs, _ = self.view_lstm(view_embs)
          view_embs = self.view_activation(self.view_proj_layer_norm(self.view_proj(view_embs[:,-1,:])))
          view_embs = self.view_activation(self.view_proj_layer_norm(self.view_proj(view_embs)))
          embs_by_view.append(view_embs)
        
        embs_by_view = torch.stack(embs_by_view, dim=1)
        # Compute View Attention
        view_attn = self.view_context_vector(embs_by_view)
        view_attention = self.view_align(self.view_context_vector(embs_by_view).squeeze(-1))
        view_attention = torch.pow(view_attention, 1 / self.T)
        view_attention = view_attention / view_attention.sum(dim=1)[:,None]
        return encoder_outputs_by_view, view_attention

    def encoder_decoder_forward(
        self,
        input_ids,
        attention_mask=None,
        view_token_idx=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs_by_view=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs_by_view is None:
            if view_token_idx is None:
                raise ValueError("If you do not pass in encoder outputs, you must pass in view_token_idx.")
            
            encoder_outputs_by_view, view_attention = self.encoder_view_forward(
                input_ids,
                view_token_idx,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we unwrap the tuple
        # then wrap the individual outputs in a BaseModelOutput when return_dict=True
        else:
            view_attention = encoder_outputs_by_view[1]
            encoder_outputs_by_view = encoder_outputs_by_view[0]
            
            for i in range(len(encoder_outputs_by_view)):
                if not isinstance(encoder_outputs_by_view[i], BaseModelOutput):
                    encoder_outputs_by_view[i] = BaseModelOutput(
                        last_hidden_state=encoder_outputs_by_view[i][0],
                        hidden_states=encoder_outputs_by_view[i][1] if len(encoder_outputs_by_view[i]) > 1 else None,
                        attentions=encoder_outputs_by_view[i][2] if len(encoder_outputs_by_view[i]) > 2 else None,
                    )
        
        # Get the last hidden state for each view.
        encoder_last_hidden_states_by_view = [eo.last_hidden_state for eo in encoder_outputs_by_view]
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states_by_view=encoder_last_hidden_states_by_view,
            encoder_attention_mask=attention_mask,
            view_attention=view_attention,
            head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            num_views=self.num_views,
        )

        if not return_dict:
            out = (decoder_outputs,)
            for encoder_output in encoder_outputs_by_view:
              out += (encoder_output,)
            return out
        
        encoder_hidden_states_by_view = [eo.hidden_states for eo in encoder_outputs_by_view]
        encoder_attentions_by_view = [eo.attentions for eo in encoder_outputs_by_view]
        return Seq2SeqMultiViewModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_last_hidden_states_by_view,
            encoder_hidden_states=encoder_hidden_states_by_view,
            encoder_attentions=encoder_attentions_by_view,
            view_attention=view_attention
        )


    def forward(
        self,
        input_ids,
        attention_mask=None,
        view_token_idx=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs_by_view=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if labels is not None:
            dedup_indices = list(range(0, labels.shape[0], self.num_views))
            labels = labels[dedup_indices,:]
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
            else:
                decoder_input_ids = decoder_input_ids[dedup_indices,:]
        
        outputs = self.encoder_decoder_forward(
            input_ids,
            attention_mask=attention_mask,
            view_token_idx=view_token_idx,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs_by_view=encoder_outputs_by_view,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqMultiViewLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            view_attention=outputs.view_attention,
        )
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs_by_view=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs_by_view": encoder_outputs_by_view,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }


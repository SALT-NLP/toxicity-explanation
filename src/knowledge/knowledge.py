from torch import nn, torch
from transformers import BertForSequenceClassification,BartModel,BartPretrainedModel
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput,Seq2SeqModelOutput,BaseModelOutput
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from inspect import signature
from tqdm import tqdm

import math

class BartForConditionalGenerationKnowledgeModel(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight", r"v"]

    def __init__(self, config, knowledge_dropout=0.0, knowledge_embed_size=600, use_cuda=True):
      super().__init__(config)
      self.model = BartModel(config)
      self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
      self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
      self.init_weights()

      hidden_size = self.config.d_model
      self.knowledge_dropout = nn.Dropout(p=knowledge_dropout)
      self.attn_softmax = nn.Softmax(dim=1)

      self.W1 = nn.Linear(hidden_size, hidden_size)
      self.W2 = nn.Linear(knowledge_embed_size, hidden_size)
      self.W3 = nn.Linear(knowledge_embed_size, hidden_size)
      self.W4 = nn.Linear(2*hidden_size, hidden_size)

    def get_encoder(self):
      return self.model.get_encoder()

    def get_decoder(self):
      return self.model.get_decoder()

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
      """
      Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
      generate method.
      """
      input_ids = decoder_input_ids.clone()
      # cut decoder_input_ids if past is used
      if past is not None:
        decoder_input_ids = decoder_input_ids[:, -1:]

      param_dict = {
          "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
          "encoder_outputs": encoder_outputs,
          "past_key_values": past,
          "decoder_input_ids": decoder_input_ids,
          "attention_mask": attention_mask,
          "head_mask": head_mask,
          "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
      }
      params = signature(self.forward).parameters
      for arg,val in kwargs.items():
        if arg in params:
          param_dict[arg] = val
      return param_dict

    def encoder_knowledge_forward(
        self,
        input_ids,
        knowledge_embeds,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
      # Encoder Forward
      encoder_outputs = self.model.encoder(
          input_ids=input_ids,
          attention_mask=attention_mask,
          head_mask=head_mask,
          inputs_embeds=inputs_embeds,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
      )
      query = self.knowledge_dropout(self.W1(encoder_outputs.last_hidden_state))
      key = self.knowledge_dropout(self.W2(knowledge_embeds))
      value = self.knowledge_dropout(self.W3(knowledge_embeds))

      sqrt_d = math.sqrt(self.config.d_model)
      out = torch.bmm(query, torch.transpose(key, 1, 2))
      out = torch.bmm(self.attn_softmax(out / sqrt_d), value)
      
      out = torch.cat((out, encoder_outputs.last_hidden_state), dim=2)
      out = self.knowledge_dropout(self.W4(out))

      encoder_outputs.last_hidden_state = out
      return encoder_outputs

    def encoder_decoder_forward(
        self,
        input_ids,
        attention_mask=None,
        knowledge_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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
      
      # Encoder
      if encoder_outputs is None:
        if knowledge_embeds is None:
          raise ValueError("If you do not pass in encoder outputs, you must pass in classifier attention.")
        encoder_outputs = self.encoder_knowledge_forward(
            input_ids,
            knowledge_embeds=knowledge_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
      # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
      elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )
      
      # Decoder
      # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
      decoder_outputs = self.model.decoder(
          input_ids=decoder_input_ids,
          attention_mask=decoder_attention_mask,
          encoder_hidden_states=encoder_outputs[0],
          encoder_attention_mask=attention_mask,
          head_mask=decoder_head_mask,
          encoder_head_mask=head_mask,
          past_key_values=past_key_values,
          inputs_embeds=decoder_inputs_embeds,
          use_cache=use_cache,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
      )
      
      if not return_dict:
        return decoder_outputs + encoder_outputs

      return Seq2SeqModelOutput(
          last_hidden_state=decoder_outputs.last_hidden_state,
          past_key_values=decoder_outputs.past_key_values,
          decoder_hidden_states=decoder_outputs.hidden_states,
          decoder_attentions=decoder_outputs.attentions,
          cross_attentions=decoder_outputs.cross_attentions,
          encoder_last_hidden_state=encoder_outputs.last_hidden_state,
          encoder_hidden_states=encoder_outputs.hidden_states,
          encoder_attentions=encoder_outputs.attentions,
      )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        knowledge_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
      # Housekeeping
      return_dict = return_dict if return_dict is not None else self.config.use_return_dict

      if labels is not None:
        if decoder_input_ids is None:
          decoder_input_ids = shift_tokens_right(
              labels, self.config.pad_token_id, self.config.decoder_start_token_id
          )
      
      outputs = self.encoder_decoder_forward(
          input_ids,
          attention_mask=attention_mask,
          knowledge_embeds=knowledge_embeds,
          decoder_input_ids=decoder_input_ids,
          encoder_outputs=encoder_outputs,
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
      
      return Seq2SeqLMOutput(
          loss=masked_lm_loss,
          logits=lm_logits,
          past_key_values=outputs.past_key_values,
          decoder_hidden_states=outputs.decoder_hidden_states,
          decoder_attentions=outputs.decoder_attentions,
          cross_attentions=outputs.cross_attentions,
          encoder_last_hidden_state=outputs.encoder_last_hidden_state,
          encoder_hidden_states=outputs.encoder_hidden_states,
          encoder_attentions=outputs.encoder_attentions,
      )

    @staticmethod
    def _reorder_cache(past, beam_idx):
      reordered_past = ()
      for layer_past in past:
        # cached cross_attention states don't have to be reordered -> they are always the same
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
        )
      return reordered_past


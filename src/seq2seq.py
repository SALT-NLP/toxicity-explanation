from torch import nn, torch
from transformers import BertForSequenceClassification
from transformers import BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from inspect import signature
from tqdm import tqdm

class BartForConditionalGenerationJoinModel(BartForConditionalGeneration):
  def __init__(self, config, join_dropout=0.0, classifiers=None):
    super(BartForConditionalGenerationJoinModel, self).__init__(config)

    # BART Seq2Seq Initializations
    if torch.cuda.is_available():
      self = self.cuda()

    self.classifiers = []
    for classifier in tqdm(classifiers):
      model = BertForSequenceClassification.from_pretrained(classifier)
      if torch.cuda.is_available():
        model = model.cuda()
      self.classifiers.append(model)
    self.classification_hidden = self.classifiers[0].config.num_hidden_layers
    self.classification_heads = self.classifiers[0].config.num_attention_heads

    # Join Embedding (dim_v = 768 (BART default hidden size))
    num_v = len(self.classifiers)
    dim_v = self.config.d_model
    self.v = nn.Parameter(torch.randn(num_v, dim_v), requires_grad=True)
    self.join_dropout = nn.Dropout(p=join_dropout)

  #def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
  def prepare_inputs_for_generation(self, input_ids, **kwargs):
    """
    Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
    generate method.
    """
    params = signature(self.forward).parameters
    param_dict = {'input_ids': input_ids}
    for arg,val in kwargs.items():
      if arg in params:
        param_dict[arg] = val
    return param_dict

  def _get_enrichment(self, classifier_inputs, classifier_attention, batch_size):
    # Compute first classifier attentions
    output = self.classifiers[0](
        input_ids=classifier_inputs,
        attention_mask=classifier_attention,
        output_attentions=True
    )
    attn_layers = output.attentions[-1].mean(dim=2)

    # Compute remaining classifier attentions
    for i,classifier in enumerate(self.classifiers[1:]):
      start = (i + 1) * self.classification_heads
      end = start + self.classification_heads

      output = classifier(
          input_ids=classifier_inputs,
          attention_mask=classifier_attention,
          output_attentions=True
      )

      attn_layers = torch.cat(
          (attn_layers, output.attentions[-1].mean(dim=2)),
          dim=1
      )

    # Apply dropout to join embedding and reshape attention prob
    v_dropout = self.join_dropout(self.v)
    attn_layers = torch.reshape(
        attn_layers,
        (batch_size, len(self.classifiers), self.classification_heads, -1)
    )

    # Elementwise multiply Attention and Join embedding and Sum
    return (attn_layers[:,:,:,:,None] * v_dropout[:,None,None,:]).sum(axis=(1,2))

  def _add_enrichment_to_beam(self, encoder_outputs, enrichment, batch_size, curr_input_size):
      encoder_hidden_size = encoder_outputs.last_hidden_state.shape[2]
      input_size = encoder_outputs.last_hidden_state.shape[1]
      num_beams = curr_input_size // batch_size

      encoder_outputs.last_hidden_state = torch.reshape(
          input=encoder_outputs.last_hidden_state,
          shape=(batch_size, num_beams, input_size, encoder_hidden_size)
      )
      encoder_outputs.last_hidden_state[:,:input_size] = enrichment[:,None,:,:] + encoder_outputs.last_hidden_state[:,:input_size]
      encoder_outputs.last_hidden_state = torch.reshape(
          input=encoder_outputs.last_hidden_state,
          shape=(-1, input_size, encoder_hidden_size)
      )

  def forward(
      self,
      input_ids,
      classifier_inputs,
      attention_mask=None,
      classifier_attention=None,
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
      beam_search=False,
  ):
    # Housekeeping
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    print(input_ids)
    print(classifier_inputs)
    print(head_mask)

    # Encode input
    if encoder_outputs is None:
      encoder_outputs = self.get_encoder()(
          input_ids=input_ids,
          attention_mask=attention_mask,
          #head_mask=head_mask,
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

    # Enrich Hidden State with Classifier Attention
    batch_size = classifier_inputs.shape[0]
    enrichment = self._get_enrichment(classifier_inputs, classifier_attention, batch_size)
    
    # Beam Search can only be true during model generation.
    if beam_search:
      print('Input Ids: ', input_ids.shape)
      curr_input_size = input_ids.shape[0]
      self._add_enrichment_to_beam(encoder_outputs, enrichment, batch_size, curr_input_size)
    # In the latter case, we simply add directly to hidden states
    else:
      encoder_outputs.last_hidden_state += enrichment

    # Decode Output
    if decoder_input_ids is None:
      if labels is not None:
        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )
      elif decoder_inputs_embeds is None:
        decoder_input_ids = shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )
      decoder_attention_mask = decoder_input_ids != self.config.pad_token_id
    
    decoder_outputs = self.get_decoder()(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
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

    lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + \
                    self.final_logits_bias

    masked_lm_loss = None
    if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            lm_logits.view(-1, self.config.vocab_size),
            labels.view(-1)
        )

    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    return Seq2SeqLMOutput(
        loss=masked_lm_loss,
        logits=lm_logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


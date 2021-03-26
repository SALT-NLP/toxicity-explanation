from torch import nn, torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput

class JoinModel(nn.Module):
  def __init__(self, classifiers, classifier_tokenizer, seq2seq_model, seq2seq_tokenizer):
    super(JoinModel, self).__init__()

    # BART Seq2Seq Initializations
    self.seq2seq_tokenizer = BartTokenizer.from_pretrained(seq2seq_tokenizer)
    self.seq2seq_model = BartForConditionalGeneration.from_pretrained(seq2seq_model)
    self.encoder = self.seq2seq_model.get_encoder()
    self.decoder = self.seq2seq_model.get_decoder()

    # BERT Classifier Initializations
    self.classifier_tokenizer = BertTokenizer.from_pretrained(classifier_tokenizer)
    self.classifiers = []
    for classifier in classifiers:
      model = BertForSequenceClassification.from_pretrained(classifier)
      self.classifiers.append(model)
    self.classification_hidden = self.classifiers[0].config.num_hidden_layers
    self.classification_heads = self.classifiers[0].config.num_attention_heads

    # Join Embedding
    num_v = 4 * self.classification_heads
    dim_v = self.seq2seq_model.config.d_model
    self.v = nn.Parameter(torch.randn(num_v, dim_v), requires_grad=True)

    # Misc.
    self.max_length = 128

  def __inference_forward(self, post):
    raise NotImplementedError

  def forward(self, post, target=None):
    if target is None:
      return self.__inference_forward(post)

    # Encode input
    encoder_inputs = self.seq2seq_tokenizer(
        post,
        return_tensors='pt',
        padding="max_length",
        max_length=self.max_length
    )
    labels = self.seq2seq_tokenizer(
        target,
        return_tensors='pt',
        padding="max_length",
        max_length=self.max_length
    ).input_ids
    encoder_outputs = self.encoder(**encoder_inputs)
    encoder_hidden_size = encoder_outputs.last_hidden_state.shape[-1]
    input_size = encoder_inputs.input_ids.shape[-1]

    # Classify Input and get Attention
    classifier_inputs = self.classifier_tokenizer(
        post,
        return_tensors='pt',
        padding="max_length",
        max_length=self.max_length)
    attn_layers = torch.empty(4 * self.classification_heads, input_size)
    for i,classifier in enumerate(self.classifiers):
      start = i * self.classification_heads
      end = start + self.classification_heads

      output = classifier(**classifier_inputs, output_attentions=True)
      attn_layers[start:end] = output.attentions[-1].mean(dim=2)

    enrichment = (attn_layers[:,:,None] * self.v[:,None,:]).sum(axis=0)
    enriched_hidden_state = enrichment + encoder_outputs.last_hidden_state
    print(enriched_hidden_state.shape)

    decoder_input_ids = labels
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=enriched_hidden_state
    )

    lm_logits = self.seq2seq_model.lm_head(decoder_outputs[0]) + \
                    self.seq2seq_model.final_logits_bias

    masked_lm_loss = None
    if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            lm_logits.view(-1, self.seq2seq_model.config.vocab_size),
            labels.view(-1)
        )

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


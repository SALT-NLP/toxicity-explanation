from transformers.generation_utils import GenerationMixin
from transformers.file_utils import ModelOutput
from transformers.generation_beam_search import BeamSearchScorer
from transformers.generation_stopping_criteria import validate_stopping_criteria
from torch.nn import functional as F
import torch

class MultiViewGenerationMixin(GenerationMixin):
    @torch.no_grad()
    def generate(
        self,
        input_ids = None,
        max_length = None,
        min_length = None,
        do_sample = None,
        early_stopping = None,
        num_beams = None,
        temperature = None,
        top_k = None,
        top_p = None,
        repetition_penalty = None,
        bad_words_ids = None,
        bos_token_id = None,
        pad_token_id = None,
        eos_token_id = None,
        length_penalty = None,
        no_repeat_ngram_size = None,
        encoder_no_repeat_ngram_size = None,
        num_return_sequences = None,
        max_time = None,
        decoder_start_token_id = None,
        use_cache = None,
        num_beam_groups = None,
        diversity_penalty = None,
        prefix_allowed_tokens_fn = None,
        output_attentions = None,
        output_hidden_states = None,
        output_scores = None,
        return_dict_in_generate = None,
        forced_bos_token_id = None,
        forced_eos_token_id = None,
        remove_invalid_values = None,
        num_views=6,
        **model_kwargs,
    ):
        # set init values
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        max_length = max_length if max_length is not None else self.config.max_length
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )
        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        # Storing encoder_input_ids for logits_processor that could use them
        encoder_input_ids = input_ids

        # set input_ids as decoder_input_ids
        if "decoder_input_ids" in model_kwargs:
            input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id, num_views=num_views
            )

        if "encoder_outputs_by_view" not in model_kwargs:
            raise ValueError("Make sure that `model_kwargs` include `encoder_outputs_by_view` of type `ModelOutput`.")
        
        for eo in model_kwargs["encoder_outputs_by_view"][0]:
            if not isinstance(eo, ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs_by_view` of type `ModelOutput`.")
        
        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # determine generation mode
        is_greedy_gen_mode = (num_beams == 1)
        is_beam_gen_mode = (num_beams > 1)

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache
        
        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=encoder_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
        )

        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length,
            max_time=max_time,
        )

        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, num_views=num_views, **model_kwargs
            )
            
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )
    
    def beam_search(
        self,
        input_ids,
        beam_scorer,
        logits_processor = None,
        stopping_criteria = None,
        max_length = None,
        pad_token_id = None,
        eos_token_id = None,
        output_attentions = None,
        output_hidden_states = None,
        output_scores = None,
        return_dict_in_generate = None,
        **model_kwargs,
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = max_length if max_length is not None else self.config.max_length
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = () if output_attentions else None
            encoder_hidden_states = () if output_hidden_states else None
            for eo in model_kwargs["encoder_outputs_by_view"][0]:
              if output_attentions:
                encoder_attentions += (eo.get("attentions"),)
              if output_hidden_states:
                encoder_hidden_states += (eo.get("hidden_states"),)

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `F.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            cur_len = cur_len + 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
    
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids,
        expand_size = 1,
        is_encoder_decoder = False,
        attention_mask = None,
        encoder_outputs_by_view = None,
        num_views=6,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        
        input_ids = input_ids.index_select(0, expanded_return_idx)
        
        if attention_mask is not None:
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.reshape(-1, num_views * seq_len)
            attention_mask = attention_mask.index_select(0, expanded_return_idx)
            model_kwargs["attention_mask"] = attention_mask.reshape(-1, seq_len)

        if is_encoder_decoder:
            assert encoder_outputs_by_view is not None
            encoder_outputs_by_view = list(encoder_outputs_by_view)
            for eo in encoder_outputs_by_view[0]:
                eo["last_hidden_state"] = eo.last_hidden_state.index_select(
                    0, expanded_return_idx.to(eo.last_hidden_state.device)
                )
            encoder_outputs_by_view[1] = encoder_outputs_by_view[1].index_select(0, expanded_return_idx)
            model_kwargs["encoder_outputs_by_view"] = tuple(encoder_outputs_by_view)
        return input_ids, model_kwargs
    
    def _prepare_decoder_input_ids_for_generation(
        self, input_ids, decoder_start_token_id = None, bos_token_id = None, num_views = 6
    ):
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
            torch.ones((input_ids.shape[0] // num_views, 1), dtype=torch.long, device=input_ids.device) * decoder_start_token_id
        )
        return decoder_input_ids


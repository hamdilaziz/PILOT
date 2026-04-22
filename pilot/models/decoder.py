from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import MBartConfig, MBartForCausalLM
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithCrossAttentions


__all__ = ["PILOTDecoder"]


class PILOTDecoder(nn.Module):
    """
    PILOT autoregressive decoder based on mBART.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.__name__ = "PILOTDecoder"

        tokenizer = params["tokenizer"]
        if tokenizer is None:
            raise ValueError("A tokenizer must be provided.")
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("The tokenizer must define pad_token_id.")

        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.extra_vocab_slots = int(params.get("extra_vocab_slots", 1))

        base_vocab_size = int(len(self.tokenizer.get_vocab()))
        self.vocab_size = base_vocab_size + self.extra_vocab_slots

        config = MBartConfig(
            is_decoder=True,
            is_encoder_decoder=bool(params.get("is_encoder_decoder", True)),
            add_cross_attention=bool(params.get("add_cross_attention", True)),
            decoder_layers=int(params.get("num_layers", 4)),
            max_position_embeddings=int(params["max_length"]),
            vocab_size=self.vocab_size,
            scale_embedding=True,
            add_final_layer_norm=True,
            pad_token_id=self.pad_token_id,
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            decoder_start_token_id=getattr(
                tokenizer,
                "bos_token_id",
                getattr(tokenizer, "cls_token_id", self.pad_token_id),
            ),
        )

        self.model = MBartForCausalLM(config)
        self.model.model.decoder.embed_tokens.padding_idx = self.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

        extra_special_tokens = params.get("extra_special_tokens")
        if extra_special_tokens:
            self.add_special_tokens(extra_special_tokens)

    def add_special_tokens(self, tokens: Iterable[str]) -> None:
        """
        Add extra special tokens and resize embeddings if needed.
        """
        unique_tokens = sorted(set(tokens))
        if not unique_tokens:
            return

        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": unique_tokens}
        )
        if num_added <= 0:
            return

        new_vocab_size = len(self.tokenizer) + self.extra_vocab_slots
        self.model.resize_token_embeddings(new_vocab_size)
        self.vocab_size = new_vocab_size
        self.model.config.vocab_size = new_vocab_size
        self.model.model.decoder.embed_tokens.padding_idx = self.pad_token_id

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[Any] = None,
        past: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for Hugging Face generation.
        """
        del kwargs

        if past is not None and past_key_values is None:
            past_key_values = past

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        if attention_mask is None:
            if input_ids is None:
                raise ValueError(
                    "attention_mask must be provided when using inputs_embeds."
                )
            attention_mask = input_ids.ne(self.pad_token_id).long()

        if past_key_values is not None:
            if input_ids is not None:
                input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:, :]

        encoder_hidden_states = None
        if encoder_outputs is not None:
            if isinstance(encoder_outputs, BaseModelOutput):
                encoder_hidden_states = encoder_outputs.last_hidden_state
            elif isinstance(encoder_outputs, tuple):
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs

        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Any] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        token_prompts: Optional[Tensor] = None,
        encoder_key_bias: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithCrossAttentions:
        # Legacy arguments are accepted for compatibility but not used.
        del token_prompts, encoder_key_bias

        loss_labels = None
        if labels is not None:
            loss_labels = labels.masked_fill(labels == self.pad_token_id, -100)

        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            labels=loss_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Optional[Tensor] = None,
        **generate_kwargs: Any,
    ) -> Tensor:
        """
        Convenience wrapper around Hugging Face generation.
        """
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.pad_token_id).long(),
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            **generate_kwargs,
        )

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        """
        Load both new PILOT decoder checkpoints and older wrapper-based checkpoints.
        """
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if new_key.startswith("decoder."):
                new_key = new_key[len("decoder."):]
            cleaned_state_dict[new_key] = value

        return super().load_state_dict(cleaned_state_dict, strict=strict)

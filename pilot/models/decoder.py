from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import MBartConfig, MBartForCausalLM
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


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

        extra_special_tokens = params.get("extra_special_tokens")
        if extra_special_tokens:
            self.add_special_tokens(extra_special_tokens)

    def add_special_tokens(self, tokens: Iterable[str]) -> int:
        """
        Add extra special tokens and resize embeddings if needed.
        """
        unique_tokens = list(dict.fromkeys(tokens))
        if not unique_tokens:
            return 0

        old_vocab = self.tokenizer.get_vocab()
        new_tokens = [token for token in unique_tokens if token not in old_vocab]
        pieces = {
            token: self.tokenizer.encode(token, add_special_tokens=False)
            for token in new_tokens
        }

        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": unique_tokens},
            replace_additional_special_tokens=False,
        )
        if num_added <= 0:
            return 0

        new_vocab_size = len(self.tokenizer) + self.extra_vocab_slots
        self.model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
        self.vocab_size = new_vocab_size
        self.model.config.vocab_size = new_vocab_size
        self.model.model.decoder.embed_tokens.padding_idx = self.pad_token_id

        # Initialize new tags from their old subword pieces, including a reused
        # legacy spare row. Existing vocabulary rows keep their pretrained values.
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings().weight
            head = self.model.get_output_embeddings().weight
            old_ids = set(old_vocab.values()) - set(self.tokenizer.all_special_ids)
            for token in new_tokens:
                ids = [i for i in pieces[token] if i in old_ids]
                if ids:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    embeddings[token_id].copy_(embeddings[ids].mean(dim=0))
                    if head.data_ptr() != embeddings.data_ptr():
                        head[token_id].copy_(head[ids].mean(dim=0))
        return int(num_added)

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
        use_cache = bool(generate_kwargs.setdefault("use_cache", True))
        if use_cache and generate_kwargs.get("past_key_values") is None:
            generate_kwargs["past_key_values"] = EncoderDecoderCache(
                DynamicCache(config=self.model.config), DynamicCache(config=self.model.config)
            )

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.pad_token_id).long(),
            encoder_hidden_states=encoder_hidden_states,
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

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import autocast

from .models.encoder import PILOTEncoder
from .models.decoder import PILOTDecoder


class PILOTModel(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        self.encoder = PILOTEncoder(config["encoder"])
        self.decoder = PILOTDecoder(config["decoder"])
        self.use_2d_positional_encoding = bool(
            config.get("use_2d_positional_encoding", True)
        )

    def encode(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    @staticmethod
    def _build_2d_positional_encoding(
        height: int,
        width: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Build the same 2D sinusoidal positional encoding used before flattening
        the CNN feature map for the decoder.
        """
        if dim % 4 != 0:
            raise ValueError(
                f"2D positional encoding requires dim % 4 == 0, got dim={dim}."
            )

        pe = torch.zeros((1, dim, height, width), device=device, dtype=dtype)

        half_dim = dim // 2
        div = torch.exp(
            -torch.arange(0, half_dim, 2, device=device, dtype=torch.float32)
            / dim
            * math.log(10000.0)
        ).to(dtype=dtype).unsqueeze(1)

        h_pos = torch.arange(0, height, device=device, dtype=torch.float32).unsqueeze(0)
        w_pos = torch.arange(0, width, device=device, dtype=torch.float32).unsqueeze(0)

        pe[:, :half_dim:2, :, :] = (
            torch.sin(div * h_pos).unsqueeze(0).unsqueeze(3).expand(1, -1, -1, width)
        )
        pe[:, 1:half_dim:2, :, :] = (
            torch.cos(div * h_pos).unsqueeze(0).unsqueeze(3).expand(1, -1, -1, width)
        )
        pe[:, half_dim::2, :, :] = (
            torch.sin(div * w_pos).unsqueeze(0).unsqueeze(2).expand(1, -1, height, -1)
        )
        pe[:, half_dim + 1::2, :, :] = (
            torch.cos(div * w_pos).unsqueeze(0).unsqueeze(2).expand(1, -1, height, -1)
        )

        return pe

    def _add_2d_positional_encoding(self, features: Tensor) -> Tensor:
        if not self.use_2d_positional_encoding:
            return features
        if features.dim() != 4:
            return features

        _, c, h, w = features.shape
        pe = self._build_2d_positional_encoding(
            height=h,
            width=w,
            dim=c,
            device=features.device,
            dtype=features.dtype,
        )
        return features + pe

    @staticmethod
    def _flatten_encoder_outputs(features: Tensor) -> Tensor:
        """
        Convert encoder outputs to (B, N, C) for decoder cross-attention.
        """
        if features.dim() == 4:
            return torch.flatten(features, start_dim=2).permute(0, 2, 1)
        if features.dim() == 3:
            return features
        raise ValueError(
            f"Unsupported encoder output shape {tuple(features.shape)}. "
            "Expected a 4D feature map or a 3D sequence."
        )

    def _encode_for_decoder(self, images: Tensor) -> Tensor:
        features = self.encoder(images)
        features = self._add_2d_positional_encoding(features)
        encoder_hidden_states = self._flatten_encoder_outputs(features)

        if images.device.type != "cuda":
            encoder_hidden_states = encoder_hidden_states.to(torch.float32)

        return encoder_hidden_states

    def _build_decoder_input_ids(
        self,
        batch_size: int,
        start_token_id: int,
        token_prompts: Optional[list[Tensor]] = None,
    ) -> tuple[Tensor, Optional[list[int]]]:
        """
        Build decoder input ids for generation.

        If token prompts are provided, sequences are left-padded:
            [pad, pad, <s>, p1, p2, ...]
        Otherwise the decoder starts from:
            [<s>]
        """
        device = next(self.parameters()).device
        pad_token_id = self.decoder.tokenizer.pad_token_id

        if token_prompts is None:
            input_ids = torch.full(
                (batch_size, 1),
                fill_value=start_token_id,
                dtype=torch.long,
                device=device,
            )
            return input_ids, None

        prompt_lengths = [len(prompt) for prompt in token_prompts]
        max_prompt_len = max(prompt_lengths)

        rows = []
        for prompt in token_prompts:
            if torch.is_tensor(prompt):
                prompt = prompt.tolist()
            else:
                prompt = list(prompt)

            row = (
                [pad_token_id] * (max_prompt_len - len(prompt))
                + [start_token_id]
                + prompt
            )
            rows.append(row)

        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        return input_ids, prompt_lengths

    def forward(
        self,
        images: Tensor,
        input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ):
        encoder_hidden_states = self._encode_for_decoder(images)

        return self.decoder(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        images: Tensor,
        input_ids: Tensor,
        encoder_attention_mask: Optional[Tensor] = None,
        **generate_kwargs: Any,
    ) -> Tensor:
        encoder_hidden_states = self._encode_for_decoder(images)

        return self.decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **generate_kwargs,
        )

    @torch.no_grad()
    def predict(
        self,
        batch_data: Dict[str, Any],
        start_token_id: Optional[int] = None,
        max_length: Optional[int] = None,
        use_amp: bool = False,
        num_beams: int = 1,
        do_sample: bool = False,
        repetition_penalty: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Simple batch inference without metric computation.

        Expected batch_data keys:
            - "imgs": Tensor of shape (B, C, H, W)
            - "token_prompt": list of token id sequences
        """
        self.eval()

        device = next(self.parameters()).device
        x = batch_data["imgs"].to(device)
        batch_size = x.size(0)

        tokenizer = self.decoder.tokenizer
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        unk_token_id = getattr(tokenizer, "unk_token_id", None)

        if start_token_id is None:
            start_token_id = self.decoder.model.config.decoder_start_token_id
        if start_token_id is None:
            raise ValueError(
                "start_token_id is None. Please provide it explicitly or set "
                "decoder_start_token_id in the decoder config."
            )

        if max_length is None:
            max_length = self.decoder.model.config.max_position_embeddings

        token_prompts = batch_data.get("token_prompt")
        decoder_input_ids, _ = self._build_decoder_input_ids(
            batch_size=batch_size,
            start_token_id=start_token_id,
            token_prompts=token_prompts,
        )

        start_time = time.time()

        with autocast(enabled=use_amp and device.type == "cuda"):
            encoder_hidden_states = self._encode_for_decoder(x)
            bad_words_ids = [[unk_token_id]] if unk_token_id is not None else None

            sequences = self.decoder.generate(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                max_length=max_length,
                early_stopping=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
                repetition_penalty=repetition_penalty,
            )

        process_time = time.time() - start_time

        prefix_len = decoder_input_ids.size(1)
        generated_only = sequences[:, prefix_len:]

        token_ids = []
        str_pred = []

        for seq in generated_only:
            seq = seq.tolist()

            if eos_token_id is not None and eos_token_id in seq:
                seq = seq[:seq.index(eos_token_id)]

            seq = [tok for tok in seq if tok != pad_token_id]
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            token_ids.append(seq_tensor)
            str_pred.append(tokenizer.decode(seq_tensor, skip_special_tokens=False))

        prompts = None
        if token_prompts is not None:
            prompts = []
            for prompt in token_prompts:
                if torch.is_tensor(prompt):
                    prompt = prompt.tolist()
                prompts.append(tokenizer.decode(prompt, skip_special_tokens=False))

        return {
            "nb_samples": batch_size,
            "str_pred": str_pred,
            "prompts": prompts,
            "token_ids": token_ids,
            "time": process_time,
        }
# ICAE that supports multi span concat

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from peft import (
    get_peft_model,
)
import math


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Condense-AI/Mistral-7B-Instruct-v0.2")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    train: bool = field(
        default=False,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    checkpoint_path: str = field(
        default="checkpoints/mistral_7b_ft_icae.safetensors",
        metadata={"help": "Checkpoint path"},
    )


@dataclass
class AdditionalArguments:
    port: int = field(default=8080, metadata={"help": "Port to run the server on"})
    devices: str = field(
        default="auto", metadata={"help": "Device type to use (e.g., cuda, cpu)"}
    )
    workers_per_device: int = field(
        default=1, metadata={"help": "Number of workers per device"}
    )
    timeout: int = field(default=60, metadata={"help": "Request timeout in seconds"})
    api_path: str = field(
        default="/condense", metadata={"help": "Path for the API endpoint"}
    )
    logging_level: str = field(default="INFO", metadata={"help": "Logging level"})
    accelerator: str = field(
        default="auto", metadata={"help": "Type of accelerator to use"}
    )
    max_batch_size: int = field(default=1, metadata={"help": "Maximum batch size"})


class ICAE(torch.nn.Module):
    def __init__(self, model_args, lora_config):
        super().__init__()
        self.device = "cuda"
        self.model_args = model_args
        self.model_name = model_args.model_name_or_path
        self.icae = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(torch.bfloat16),
            resume_download=True,
        )
        self.icae.to(self.device)

        self.vocab_size = self.icae.config.vocab_size + 1  # [PAD] token
        self.pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = 4

        # tunable
        self.mem_size = 128
        self.vocab_size_with_mem = (
            self.vocab_size + self.mem_size
        )  # so, the mem tokens are in the range [self.vocab_size, self.vocab_size + self.mem_size)

        # special tokens in addition to mem and length tokens
        self.ae_token_id = self.vocab_size_with_mem + 0
        self.lm_token_id = self.vocab_size_with_mem + 1
        self.ft_token_id = self.vocab_size_with_mem + 2

        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 3)

        # special tokens for Llama-2/Mistral tokenizer
        self.bos_id = 1
        self.eos_id = 2

        self.dim = self.icae.config.hidden_size
        self.icae = get_peft_model(self.icae, lora_config)

        self.icae.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.memory_token_embed = nn.Embedding(
            self.mem_size + 3, self.dim, padding_idx=None
        )
        self.append_sequence = torch.arange(
            self.vocab_size,
            self.vocab_size + self.mem_size,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)  # mem tokens

    def compute_num_segments(self, total_length):
        assert total_length > 0
        num_segments = math.ceil(
            total_length / (self.mem_size * self.mean_compression_rate)
        )
        return num_segments

    def tokens_to_embeddings(
        self, token_ids
    ):  # input_tokens can be either normal tokens and special tokens
        embeddings = self.icae.get_base_model().model.embed_tokens(token_ids)
        special_flags = token_ids >= self.vocab_size
        embeddings[special_flags] = self.memory_token_embed(
            token_ids[special_flags].to(self.device) - self.vocab_size.to(self.device)
        ).to(
            self.device
        )  # replace special token's embedding from self.memory_token_embed
        return embeddings

    def _compress(
        self, input_ids: torch.LongTensor = None
    ):  # for inference; compress a fixed length of input into memory slots
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)

        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((max_compressed_length, self.dim))

        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat(
                [
                    segment_input_ids.to(self.device),
                    self.append_sequence.to(self.device),
                ],
                dim=1,
            ).to(self.device)
            mem_flag = segment_input_ids >= self.vocab_size
            embed_tokens = self.icae.get_base_model().model.embed_tokens
            embed_tokens.to(self.device)
            segment_input_embedding = embed_tokens(segment_input_ids)
            self.memory_token_embed.to(self.device)
            segment_input_embedding[mem_flag] = self.memory_token_embed(
                segment_input_ids[mem_flag] - self.vocab_size
            ).to(segment_input_embedding)
            segment_input_embedding = segment_input_embedding.to(self.device)
            # compress the current segment
            self.icae.to(self.device)
            segment_compress_outputs = self.icae(
                inputs_embeds=segment_input_embedding, output_hidden_states=True
            )
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

            # collect memory tokens
            compress_outputs[
                segment_idx * self.mem_size : self.mem_size * (segment_idx + 1)
            ] = segment_compress_outputs[mem_flag]

            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()

        return compress_outputs

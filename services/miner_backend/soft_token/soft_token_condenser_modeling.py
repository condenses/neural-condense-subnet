import torch
import torch.nn as nn
import huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer


class Condenser(nn.Module):
    """
    A neural module for condensing large text contexts into smaller dense representations
    """

    def __init__(
        self,
        num_condense_tokens: int,
        hidden_size: int,
        n_last_hidden_states: int,
        condense_model: AutoModelForCausalLM,
        condense_tokenizer: AutoTokenizer,
    ):
        super().__init__()
        self.dtype = torch.bfloat16
        self.num_condense_tokens = num_condense_tokens
        self.hidden_size = hidden_size
        self.n_last_hidden_states = n_last_hidden_states
        self.condense_model = condense_model
        self.condense_tokenizer = condense_tokenizer

        self.norm = nn.LayerNorm(self.hidden_size * n_last_hidden_states).to(
            dtype=self.dtype, device="cuda"
        )
        self.linear = nn.Linear(self.hidden_size * n_last_hidden_states, 4096).to(
            dtype=self.dtype, device="cuda"
        )
        self.pre_condensed_tokens = nn.Parameter(
            torch.randn(
                1,
                num_condense_tokens,
                self.hidden_size,
                dtype=self.dtype,
                device="cuda",
            )
        )

    @classmethod
    def from_pretrained(cls, repo_id, local_dir="./", dtype=torch.bfloat16):
        # Download and load checkpoint
        file_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename="checkpoints/modules.pt",
            local_dir=local_dir,
        )
        state_dict = torch.load(file_path)

        # Extract model configuration
        num_condense_tokens = state_dict["modules"]["pre_condensed_tokens"].shape[1]
        hidden_size = state_dict["modules"]["pre_condensed_tokens"].shape[2]
        linear_input_dim = state_dict["modules"]["linear_state_dict"]["weight"].shape[1]
        n_last_hidden_states = linear_input_dim // hidden_size

        # Load model and tokenizer
        condense_model = AutoModelForCausalLM.from_pretrained(
            repo_id, torch_dtype=dtype
        ).to("cuda")
        condense_tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
        condense_tokenizer.pad_token = condense_tokenizer.eos_token

        # Initialize and load state_dict
        model = cls(
            num_condense_tokens,
            hidden_size,
            n_last_hidden_states,
            condense_model,
            condense_tokenizer,
        )
        model.load_state_dict(state_dict["modules"])
        return model

    def load_state_dict(self, state_dict: dict):
        self.pre_condensed_tokens.data = state_dict["pre_condensed_tokens"].to(
            dtype=self.dtype, device="cuda"
        )
        self.linear.load_state_dict(
            {
                k: v.to(dtype=self.dtype, device="cuda")
                for k, v in state_dict["linear_state_dict"].items()
            }
        )
        self.norm.load_state_dict(
            {
                k: v.to(dtype=self.dtype, device="cuda")
                for k, v in state_dict["norm_state_dict"].items()
            }
        )

    @torch.no_grad()
    def compress(self, context: str) -> torch.Tensor:
        # Tokenize and process context
        output = self.condense_tokenizer(
            context,
            return_tensors="pt",
            add_special_tokens=False,
            padding="max_length",
            max_length=4096,
            truncation=True,
            return_attention_mask=True,
        )
        context_ids = output.input_ids.to(device="cuda")
        attention_mask = output.attention_mask.to(device="cuda")

        # Processing embedding and condensation
        context_embeds = self.condense_model.get_input_embeddings()(context_ids)
        inputs_embeds_condense = torch.cat(
            [context_embeds, self.pre_condensed_tokens], dim=1
        )
        expanded_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    self.num_condense_tokens,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

        # Generate condensed tokens
        output = self.condense_model(
            inputs_embeds=inputs_embeds_condense,
            output_hidden_states=True,
            attention_mask=expanded_attention_mask,
        )
        hidden_states = torch.cat(
            output.hidden_states[-self.n_last_hidden_states :], dim=-1
        )[:, -self.num_condense_tokens :, :]

        return self.linear(self.norm(hidden_states)).to(torch.float32)

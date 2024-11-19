import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_VALUE = 30


def perplexity(
    compressed_tokens: torch.Tensor,
    activation_prompt: str,
    expected_completion: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_tokens: int = 4096,
    **kwargs,
) -> float:
    device = model.device
    dtype = model.dtype
    completion_text = activation_prompt + expected_completion
    completion_ids = tokenizer(
        completion_text,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_tokens,
        **kwargs,
    ).input_ids.to(device=device, dtype=torch.long)
    completion_embeddings = model.get_input_embeddings()(completion_ids).to(
        dtype=dtype, device=device
    )
    compressed_tokens = compressed_tokens.to(dtype=dtype, device=device).unsqueeze(0)
    n_compressed_tokens = compressed_tokens.shape[1]
    labels = torch.cat(
        [
            torch.full((n_compressed_tokens,), -100, dtype=torch.long, device=device),
            completion_ids,
        ]
    )
    outputs = model(inputs_embeds=completion_embeddings)
    logits = outputs.logits[:, :-1, :]
    labels = labels[:, 1:]
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
    perplexity = torch.exp(loss)
    return perplexity.item()


def preprocess_batch(values: list[float]) -> list[float]:
    # Check if all values are None
    if all(value is None for value in values):
        return [DEFAULT_VALUE] * len(values)
    else:
        valid_values = [value for value in values if value is not None]
        max_value = max(valid_values)
        return [max_value * 10 if value is None else value for value in values]

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

DEFAULT_VALUE = 30


def perplexity(
    kv_cache: DynamicCache,
    activation_prompt: str,
    expected_completion: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_tokens: int = 4096,
    **kwargs,
) -> float:
    device = model.device
    completion_text = activation_prompt + expected_completion
    completion_ids = tokenizer(
        completion_text,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_tokens,
        **kwargs,
    ).input_ids.to(device=device, dtype=torch.long)
    num_seen_tokens = kv_cache._seen_tokens
    input_ids = torch.cat(
        [
            torch.full(
                (1, num_seen_tokens),
                -100,
                dtype=torch.long,
                device=device,
            ),
            completion_ids,
        ],
        dim=1,
    )
    print(input_ids.shape)
    outputs = model(input_ids=input_ids, past_key_values=kv_cache)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
        ignore_index=-100,
    )
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

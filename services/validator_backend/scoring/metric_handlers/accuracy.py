import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, DynamicCache, AutoModelForCausalLM
import structlog
from copy import deepcopy
from typing import List
from ..anti_exploitation.filter_existance import FilterExistanceChecker
from ..utils import generate_answer
from openai import OpenAI


OPENAI_CLIENT = OpenAI(base_url="https://api.together.xyz/v1")

logger = structlog.get_logger("accuracy")

DEFAULT_VALUE = 0


def accuracy(
    filter_existance_checker: FilterExistanceChecker,
    embed_model: AutoModel,
    kv_cache: DynamicCache,
    activation_prompt: str,
    expected_completion: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    positive_chunk: str,
    negative_chunk: str,
    max_tokens: int = 256,
    context: str = "",
    **kwargs,
) -> float:
    device = model.device
    context_ids = tokenizer.encode(
        context,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device=device, dtype=torch.long)
    context_length = context_ids.shape[1]
    num_seen_tokens = kv_cache._seen_tokens
    logger.debug(f"Num seen tokens: {num_seen_tokens}")
    if not filter_existance_checker.filter_existance(
        tokenizer=tokenizer,
        model=model,
        kv_cache=kv_cache,
        positive_chunk=positive_chunk,
        negative_chunk=negative_chunk,
        context_length=context_length,
    ):
        logger.warning("Existance check failed")
        return 0

    expected_completion_ids = tokenizer(
        expected_completion,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device=device, dtype=torch.long)
    n_expected_completion_tokens = expected_completion_ids.shape[1]
    max_new_tokens = int(n_expected_completion_tokens * 1.5)
    prompt_ids = tokenizer(
        activation_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_tokens,
    ).input_ids.to(device=device, dtype=torch.long)

    completion = generate_answer(
        model=model,
        tokenizer=tokenizer,
        question_ids=prompt_ids,
        cache=kv_cache,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
    )
    ground_truth = expected_completion.strip()
    logger.debug(f"Activation prompt: {activation_prompt}")
    logger.debug(f"Completion: {completion}")
    logger.debug(f"Ground truth: {ground_truth}")
    return get_accuracy_llm(completion, ground_truth, question)


def get_accuracy(completion: str, ground_truth: str, embed_model: AutoModel) -> float:
    query_instruction = (
        "Instruct: Given a text, retrieve the text that has similar meaning.\nQuery:"
    )
    queries = [ground_truth]
    passages = [completion]
    max_length = 1024

    query_embeddings = embed_model.encode(
        queries, instruction=query_instruction, max_length=max_length
    )
    passage_embeddings = embed_model.encode(
        passages, instruction="", max_length=max_length
    )
    similarity = (query_embeddings @ passage_embeddings.T) * 100
    similarity_percentage = int(similarity[0][0].item())
    if similarity_percentage < 50:
        score = 0.1
    elif similarity_percentage < 80:
        score = 0.5
    else:
        score = 1
    logger.debug(f"Score: {score}, similarity: {similarity_percentage}")
    return score


def preprocess_batch(values: list[float]) -> list[float]:
    return [value if value is not None else DEFAULT_VALUE for value in values]


def get_accuracy_llm(completion: str, ground_truth: str, question: str) -> float:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that evaluates the correctness of a response to a question based on the ground truth.",
        },
        {
            "role": "user",
            "content": f"""Please evaluate the correctness of the following response to the question based on the ground truth.\n\n**Question**: {question}\n\n**Response**: {completion}\n\n**Ground truth**: {ground_truth}
You have to return 'yes' if the response is correct, 'no' if it is incorrect. The correct response should be have same meaning as the ground truth, don't need to be exactly the same. Please just return only 'yes' or 'no', don't need to explain.
""",
        },
    ]
    response = OPENAI_CLIENT.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=messages,
        temperature=0,
        max_tokens=16,
    )
    is_correct = "yes" in response.choices[0].message.content.lower()
    return 1 if is_correct else 0.1

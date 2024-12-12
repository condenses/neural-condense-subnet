import torch
from transformers import (
    AutoTokenizer,
    DynamicCache,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)
import structlog
from ..anti_exploitation.filter_existance import FilterExistanceChecker
from ..utils import generate_answer
from ..datatypes import GroundTruthRequest
from openai import OpenAI


OPENAI_CLIENT = OpenAI(base_url="https://api.together.xyz/v1")

logger = structlog.get_logger("accuracy")

DEFAULT_VALUE = 0


def accuracy(
    filter_existance_checker: FilterExistanceChecker,
    kv_cache: DynamicCache,
    ground_truth_request: GroundTruthRequest,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    judge_pipeline: TextGenerationPipeline,
    max_tokens: int = 256,
    **kwargs,
) -> float:
    activation_prompt = ground_truth_request.activation_prompt
    expected_completion = ground_truth_request.expected_completion
    context = ground_truth_request.context
    positive_chunks = ground_truth_request.positive_chunks
    negative_chunks = ground_truth_request.negative_chunks
    device = model.device
    context_ids = tokenizer.encode(
        context,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device=device, dtype=torch.long)
    context_length = context_ids.shape[1]
    num_seen_tokens = kv_cache._seen_tokens
    logger.debug("condense-length", length=num_seen_tokens)
    if not filter_existance_checker.filter_existance(
        tokenizer=tokenizer,
        model=model,
        kv_cache=kv_cache,
        positive_chunks=positive_chunks,
        negative_chunks=negative_chunks,
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
    return get_accuracy_llm(completion, ground_truth, activation_prompt, judge_pipeline)


def preprocess_batch(values: list[float]) -> list[float]:
    return [value if value is not None else DEFAULT_VALUE for value in values]


def get_accuracy_llm(
    completion: str,
    ground_truth: str,
    question: str,
    judge_pipeline: TextGenerationPipeline,
) -> float:
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
    completion = judge_pipeline(
        messages,
        do_sample=False,
        max_new_tokens=16,
    )[0]["generated_text"][-1]["content"]
    logger.debug(f"LLM Judge Messages: {messages}")
    logger.debug(f"LLM Judge Response: {completion}")
    is_correct = "yes" in completion.lower()
    return 1 if is_correct else 0.1

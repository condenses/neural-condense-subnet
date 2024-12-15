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
import os

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("OPENAI_MODEL")
print(f"Using model: {MODEL}")

logger = structlog.get_logger("accuracy")

DEFAULT_VALUE = 0


def accuracy(
    filter_existance_checker: FilterExistanceChecker,
    kv_cache: DynamicCache,
    ground_truth_request: GroundTruthRequest,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
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
    return get_accuracy_llm(completion, ground_truth, activation_prompt)


def preprocess_batch(values: list[float]) -> list[float]:
    return [value if value is not None else DEFAULT_VALUE for value in values]


def get_accuracy_llm(
    completion: str,
    ground_truth: str,
    question: str,
) -> float:
    messages = [
        {
            "role": "system",
            "content": "You are a strict and objective evaluator tasked with scoring the correctness of a response to a question. Your job is to compare the response with the ground truth and determine if the response has the same meaning as the ground truth, regardless of exact wording.",
        },
        {
            "role": "user",
            "content": f"""
Evaluate the correctness of the provided response based on the question and the ground truth. 

Here is the information you will use for evaluation:
1. **Question**: {question}
2. **Response**: {completion}
3. **Ground Truth**: {ground_truth}

**Evaluation Criteria**:
- The response is **correct** if it conveys the same meaning as the ground truth. Minor differences in phrasing are acceptable, but the content must align fully with the intent and details of the ground truth.
- The response is **incorrect** if it:
  - Contradicts the ground truth.
  - Omits critical information from the ground truth.
  - Includes incorrect, unrelated, or fabricated information.

**Response Format**:
- Return only 'yes' if the response is correct.
- Return only 'no' if the response is incorrect.
- Do not provide explanations, reasoning, or additional commentary.

""",
        },
    ]
    completion = CLIENT.chat.completions.create(
        model=MODEL or "gpt-4o-mini",
        messages=messages,
        max_tokens=16,
    ).choices[0].message.content
    logger.debug(f"LLM Judge Messages: {messages}")
    logger.debug(f"LLM Judge Response: {completion}")
    is_correct = "yes" in completion.lower()
    return 1 if is_correct else 0.1

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
from neural_condense_core.protocol import TaskData
from copy import deepcopy

logger = structlog.get_logger("accuracy")

DEFAULT_VALUE = 0


def accuracy(
    filter_existance_checker: FilterExistanceChecker,
    kv_cache: DynamicCache,
    task_data: TaskData,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    judge_pipeline: TextGenerationPipeline,
    max_tokens: int = 256,
    **kwargs,
) -> float:
    context = task_data.formatted_context
    positive_chunks = task_data.positive_chunks
    negative_chunks = task_data.negative_chunks
    formatted_questions = task_data.formatted_questions
    questions = task_data.challenge_questions
    answers = task_data.challenge_answers

    device = model.device
    context_ids = tokenizer.encode(
        context,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device=device, dtype=torch.long)
    context_length = context_ids.shape[1]
    num_seen_tokens = kv_cache._seen_tokens
    logger.debug("condense-length", length=num_seen_tokens)
    chunk_existance_accuracy: float = filter_existance_checker.filter_existance(
        tokenizer=tokenizer,
        model=model,
        kv_cache=kv_cache,
        positive_chunks=positive_chunks,
        negative_chunks=negative_chunks,
        context_length=context_length,
    )
    if chunk_existance_accuracy <= 0.1:
        logger.info(
            f"Too low chunk existance accuracy, skipping scoring: {chunk_existance_accuracy}"
        )
        return 0

    questions_ids = [
        tokenizer(
            question,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_tokens,
        ).input_ids.to(device=device, dtype=torch.long)
        for question in formatted_questions
    ]
    accuracies = []
    for question_ids, formatted_question, answer, question in zip(
        questions_ids, formatted_questions, answers, questions
    ):
        expected_completion_ids = tokenizer(
            answer,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device=device, dtype=torch.long)
        n_expected_completion_tokens = expected_completion_ids.shape[1]
        max_new_tokens = max(int(n_expected_completion_tokens * 1.5), 8)
        _kv_cache = deepcopy(kv_cache)
        logger.debug("kv_length", length=_kv_cache._seen_tokens)
        completion = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question_ids=question_ids,
            cache=_kv_cache,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
        )
        ground_truth = answer.strip()
        logger.debug(f"Question: {formatted_question}")
        logger.debug(f"Completion: {completion}")
        logger.debug(f"Ground truth: {ground_truth}")
        accuracy = get_accuracy_llm(completion, ground_truth, question, judge_pipeline)
        accuracies.append(accuracy)
    logger.info(f"Accuracies: {accuracies}")
    return chunk_existance_accuracy * sum(accuracies) / len(accuracies)


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
    )[0][
        "generated_text"
    ][-1]["content"]
    logger.debug(f"LLM Judge Response: {completion}")
    is_correct = "yes" in completion.lower()
    return 1 if is_correct else 0

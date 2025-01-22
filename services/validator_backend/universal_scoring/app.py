from fastapi import FastAPI
import numpy as np
from together import Together
from typing import List
import logging
from pydantic import BaseModel
from neural_condense_core import logger


class TaskData(BaseModel):
    formatted_context: str = ""
    original_context: str = ""
    challenge_questions: List[str] = []
    challenge_answers: List[str] = []
    formatted_questions: List[str] = []
    negative_chunks: List[str] = []
    positive_chunks: List[str] = []


class UtilData(BaseModel):
    compressed_kv_b64: str = ""
    compressed_length: int = 0
    download_time: float = 0.0
    bonus_compress_size: float = 0.0
    bonus_time: float = 0.0
    local_filename: str = ""


class MinerResponse(BaseModel):
    filename: str = ""
    compressed_context: str = ""


class BatchedScoringRequest(BaseModel):
    miner_responses: List[MinerResponse] = []
    task_data: TaskData = TaskData()
    target_model: str = ""
    criterias: List[str] = []


# logger = logging.getLogger("uvicorn")
logger.info("This will show in Universal Validator Backend logs")

app = FastAPI()

openai_client = Together()


@app.post("/get_metrics")
async def get_metrics(item: BatchedScoringRequest):
    logger.info(f"Received scoring request for model: {item.target_model}")

    model = item.target_model
    if "Llama-3.1-8B-Instruct" in model:
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K"

    compressed_contexts = [
        item.miner_responses[i].compressed_context
        for i in range(len(item.miner_responses))
    ]
    questions = item.task_data.challenge_questions
    ground_truths = item.task_data.challenge_answers
    negative_chunks = item.task_data.negative_chunks
    positive_chunks = item.task_data.positive_chunks

    logger.info(f"Processing {len(compressed_contexts)} miner responses")
    logger.info(f"Number of questions: {len(questions)}")
    logger.info(f"Number of positive chunks: {len(positive_chunks)}")
    logger.info(f"Number of negative chunks: {len(negative_chunks)}")
    logger.info(f"Number of compressed contexts: {len(compressed_contexts)}")
    existence_scores = []
    for i, compressed_context in enumerate(compressed_contexts):
        logger.info(
            f"Getting existence score for response {i+1}/{len(compressed_contexts)}"
        )
        score = await get_chunk_existence_score(
            compressed_context, positive_chunks, negative_chunks, model
        )
        logger.info(f"Raw existence score: {score}, type: {type(score)}")
        if score is None or np.isnan(score):
            logger.error(f"Invalid existence score for response {i+1}")
            score = 0.0
        existence_scores.append(score)
        logger.info(f"Existence score for response {i+1}: {score}")

    qa_scores = []
    for i, compressed_context in enumerate(compressed_contexts):
        logger.info(f"Getting QA score for response {i+1}/{len(compressed_contexts)}")
        score = await get_qa_score(compressed_context, questions, ground_truths, model)
        logger.info(f"Raw QA score: {score}, type: {type(score)}")
        if score is None or np.isnan(score):
            logger.error(f"Invalid QA score for response {i+1}")
            score = 0.0
        qa_scores.append(score)
        logger.info(f"QA score for response {i+1}: {score}")

    final_scores = []
    for i, (existence_score, qa_score) in enumerate(zip(existence_scores, qa_scores)):
        logger.info(f"Calculating final score for response {i+1}")
        logger.info(f"Using existence_score={existence_score}, qa_score={qa_score}")
        if (
            existence_score is None
            or qa_score is None
            or np.isnan(existence_score)
            or np.isnan(qa_score)
        ):
            logger.error(f"Invalid scores for response {i+1}")
            final_score = 0.0
        else:
            final_score = (existence_score + qa_score) / 2
        final_scores.append(final_score)
        logger.info(f"Final score for response {i+1}: {final_score}")

    avg_score = np.mean(final_scores) if final_scores else 0.0
    logger.info(f"Completed scoring. Raw scores: {final_scores}")
    logger.info(f"Completed scoring. Average final score: {avg_score:.4f}")
    return {"metrics": {"accuracy": final_scores}}


async def get_qa_score(
    compressed_context: str,
    questions: List[str],
    ground_truths: List[str],
    model: str,
):
    logger.info("Starting QA scoring")
    prompt = """
You are given a context and a question. Your task is to answer the question based on the context.
---CONTEXT---
{compressed_context}
---QUESTION---
{question}
---END---
Your response should be concise and to the point. Skip any greetings or other unrelevant information.
"""
    answers = []

    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}")
        messages = [
            {
                "role": "user",
                "content": prompt.format(
                    compressed_context=compressed_context, question=question
                ),
            },
        ]
        try:
            response = openai_client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
            text = response.choices[0].message.content
            answers.append(text)
        except Exception as e:
            logger.error(f"Error getting answer for question {i+1}: {str(e)}")
            raise

    judge_prompt = """
You are given a set of question, answer, and ground truth. Your task is to determine whether the answer is correct.
---QUESTION---
{question}
---ANSWER---
{answer}
---GROUND TRUTH---
{ground_truth}
---END---
You only need to output one word: either 'yes' or 'no'. No additional text or explanations are required.
An answer is correct if it is mentioned the important points in the ground truth.
"""
    scores = []
    for i, (question, answer, ground_truth) in enumerate(
        zip(questions, answers, ground_truths)
    ):
        logger.info(f"Judging answer {i+1}/{len(questions)}")
        messages = [
            {
                "role": "user",
                "content": judge_prompt.format(
                    question=question, answer=answer, ground_truth=ground_truth
                ),
            },
        ]
        try:
            response = openai_client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
            text = response.choices[0].message.content
            text = text.strip().lower()
            words = text.split()
            result = "yes" in words and not ("no" in words or "not" in words)
            scores.append(result)
            logger.info(f"Answer {i+1} scored: {result}")
        except Exception as e:
            logger.error(f"Error judging answer {i+1}: {str(e)}")
            raise

    if not scores:
        logger.warning("No valid scores generated in QA scoring")
        return 0.0

    score = np.mean(scores)
    logger.info(f"Final QA score: {score:.4f}")
    return score


async def get_chunk_existence_score(
    compressed_context: str,
    positive_chunks: List[str],
    negative_chunks: List[str],
    model: str,
):
    logger.info("Starting chunk existence scoring")
    prompt = """
You are given a context and a chunk of text. Your task is to determine whether the chunk content is mentioned in the context.
---CONTEXT---
{compressed_context}
---CHUNK---
{chunk}
---END---
Your response should contain exactly one word: either 'yes' or 'no'. No additional text or explanations are required.
"""
    positive_scores = []
    negative_scores = []

    for i, chunk in enumerate(positive_chunks):
        logger.info(f"Processing positive chunk {i+1}/{len(positive_chunks)}")
        messages = [
            {
                "role": "user",
                "content": prompt.format(
                    compressed_context=compressed_context, chunk=chunk
                ),
            },
        ]
        try:
            response = openai_client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
            text = response.choices[0].message.content
            text = text.strip().lower()
            words = text.split()
            result = "yes" in words and not ("no" in words or "not" in words)
            positive_scores.append(result)
            logger.info(f"Positive chunk {i+1} scored: {result}")
        except Exception as e:
            logger.error(f"Error processing positive chunk {i+1}: {str(e)}")
            raise

    for i, chunk in enumerate(negative_chunks):
        logger.info(f"Processing negative chunk {i+1}/{len(negative_chunks)}")
        messages = [
            {
                "role": "user",
                "content": prompt.format(
                    compressed_context=compressed_context, chunk=chunk
                ),
            },
        ]
        try:
            response = openai_client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
            text = response.choices[0].message.content
            text = text.strip().lower()
            words = text.split()
            result = ("no" in words or "not" in words) and "yes" not in words
            negative_scores.append(result)
            logger.info(f"Negative chunk {i+1} scored: {result}")
        except Exception as e:
            logger.error(f"Error processing negative chunk {i+1}: {str(e)}")
            raise

    if not positive_scores and not negative_scores:
        logger.warning("No valid scores generated in chunk existence scoring")
        return 0.0

    score = np.mean(positive_scores + negative_scores)
    logger.info(f"Final existence score: {score:.4f}")
    return score

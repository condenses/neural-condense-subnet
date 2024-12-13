from transformers import AutoTokenizer
import re
import substrateinterface as st
from .scheduler import Scheduler
from .convo_generator import ConvoGenerator
from .schemas import QASchedulerConfig
import random
import os
from typing import Tuple
from ...protocol import TextCompressProtocol
from .filter_chunker import FilterExistanceChecker
from .utils import retry

CORCEL_API_KEY = os.getenv("CORCEL_API_KEY")
CORCEL_BASE_URL = os.getenv(
    "CORCEL_BASE_URL", "https://api.corcel.io/v1/text/vision/chat"
)
GENERATOR_MODEL_ID = os.getenv("GENERATOR_MODEL_ID", "llama-3-1-8b")


class ChallengeGenerator:
    def __init__(self, keypair: st.Keypair):
        """
        Initialize the ChallengeGenerator class with various dataset loaders and configuration tokens.
        """
        self.generator = ConvoGenerator(keypair=keypair)
        self.synthesizer = Scheduler(
            generator=self.generator,
            qa_config=QASchedulerConfig(n_qa_per_context=4, max_items=100),
            refresh_time=60.0,
        )
        self.synthesizer.start()
        self.start_activation_token = "<START-ACTIVATE-TOKEN>"
        self.end_activation_token = "<END-ACTIVATE-TOKEN>"
        self.task_to_builder = {
            "question_answering": self._build_qa_conversation,
        }
        self.filter_checker = FilterExistanceChecker(chunk_size=256)

    @retry(max_attempts=3)
    async def generate_challenge(
        self,
        tokenizer: AutoTokenizer,
        task: str = "question_answering",
        max_context_length_in_chars: int = 10000,
    ) -> TextCompressProtocol:
        try:
            context, challenge_question, challenge_answer = await self.task_to_builder[
                task
            ](max_context_length_in_chars)
            positive_chunks, negative_chunks = self.filter_checker.get_chunks(context)
            synapse = self._build_protocol(
                tokenizer, context, challenge_question, challenge_answer
            )
            synapse.positive_chunks = positive_chunks
            synapse.negative_chunks = negative_chunks
        except Exception as e:
            raise e
        return synapse

    @retry(max_attempts=3)
    async def _build_qa_conversation(self, max_chars: int) -> Tuple[str, str, str]:
        context_qa_items = await self.synthesizer.get_qas(n=10)
        context = ""
        question_answer_pairs = []
        for qa_item in context_qa_items:
            if len(context) + len(qa_item.context_seed) > max_chars:
                continue
            context += f"\n{qa_item.context_seed}"
            questions = qa_item.questions
            answers = qa_item.answers
            question_answer_pairs.extend(list(zip(questions, answers)))
        random.shuffle(question_answer_pairs)
        challenge_question, challenge_answer = question_answer_pairs.pop()

        return context, challenge_question, challenge_answer

    def _build_protocol(
        self,
        tokenizer: AutoTokenizer,
        context: str,
        challenge_question: str,
        challenge_answer: str,
    ) -> TextCompressProtocol:
        messages = [
            {
                "role": "user",
                "content": f"{context}{self.start_activation_token}\n\nRead the provided information and answer the following question, the answer should be retrieved from the provided information: {challenge_question}",
            },
            {
                "role": "assistant",
                "content": f"{self.end_activation_token}{challenge_answer}",
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        context, activation_prompt, _ = re.split(
            f"{re.escape(self.start_activation_token)}|{re.escape(self.end_activation_token)}",
            prompt,
        )
        return TextCompressProtocol(
            context=context,
            activation_prompt=activation_prompt,
            expected_completion=challenge_answer,
            messages=messages,
        )

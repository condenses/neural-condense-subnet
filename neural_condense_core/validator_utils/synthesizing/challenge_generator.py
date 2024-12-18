from transformers import AutoTokenizer
import re
import substrateinterface as st
from .scheduler import Scheduler
from .convo_generator import ConvoGenerator
from .schemas import QASchedulerConfig
import random
import os
from typing import Tuple, List
from ...protocol import TextCompressProtocol
from .filter_chunker import FilterExistanceChecker
from ...constants import constants, ChatTemplate
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
        model_name: str,
        task: str = "question_answering",
        max_context_length_in_chars: int = 10000,
    ) -> TextCompressProtocol:
        chat_template = constants.CHAT_TEMPLATES[model_name.split("/")[-1]]
        try:
            context, challenge_question, challenge_answer = await self.task_to_builder[
                task
            ](max_context_length_in_chars)
            positive_chunks, negative_chunks = self.filter_checker.get_chunks(context)
            synapse = self._build_protocol(
                chat_template,
                context,
                challenge_question,
                challenge_answer,
                positive_chunks,
                negative_chunks,
            )
        except Exception as e:
            raise e
        return synapse

    @retry(max_attempts=3)
    async def _build_qa_conversation(self, max_chars: int) -> Tuple[str, str, str]:
        context_qa_items = await self.synthesizer.get_qas(n=20)
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
        challenge_qa_pairs = random.sample(question_answer_pairs, 5)
        challenge_questions = [qa_pair[0] for qa_pair in challenge_qa_pairs]
        challenge_answers = [qa_pair[1] for qa_pair in challenge_qa_pairs]

        return context, challenge_questions, challenge_answers

    def _build_protocol(
        self,
        chat_template: ChatTemplate,
        context: str,
        challenge_questions: List[str],
        challenge_answers: List[str],
        positive_chunks: List[str],
        negative_chunks: List[str],
    ) -> TextCompressProtocol:
        formatted_context = chat_template.apply_context_template(context)
        formatted_questions = [
            chat_template.apply_question_template(question)
            for question in challenge_questions
        ]

        return TextCompressProtocol.model_validate(
            {
                "task_data": {
                    "original_context": context,
                    "challenge_questions": challenge_questions,
                    "challenge_answers": challenge_answers,
                    "formatted_questions": formatted_questions,
                    "positive_chunks": positive_chunks,
                    "formatted_context": formatted_context,
                    "negative_chunks": negative_chunks,
                },
                "context": formatted_context,
            }
        )

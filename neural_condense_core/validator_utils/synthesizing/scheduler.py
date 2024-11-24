import redis
from typing import Optional, List
import random
from datasets import load_dataset
from .convo_generator import ConvoGenerator
from .custom_dataset_loaders import load_instruct_datasets, load_context_datasets
from .schemas import (
    QASet,
    Conversation,
    Message,
    QASchedulerConfig,
    ConversationSchedulerConfig,
)
import time
from ...logger import logger
import asyncio


class Scheduler:
    def __init__(
        self,
        generator: ConvoGenerator,
        qa_config: QASchedulerConfig,
        convo_config: ConversationSchedulerConfig,
        refresh_time: float = 10.0,
    ):
        self.generator = generator
        self.qa_config = qa_config
        self.convo_config = convo_config
        self.refresh_time = refresh_time
        self.instruct_datasets = load_instruct_datasets()
        self.context_datasets = load_context_datasets()
        self.redis = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True,  # Configure as needed
        )
        self.qa_key = "qa_sets"
        self.convo_key = "conversations"
        self._prefill_queues()
        self.running = False
        self.loop = asyncio.get_event_loop()

    def _prefill_queues(self):
        self.redis.delete(self.qa_key)
        self.redis.delete(self.convo_key)
        cached_qa_ds = load_dataset(
            "Condense-AI/subnet-synthetic-dataset-v0.2", name="QA", split="train"
        )
        cached_convo_ds = load_dataset(
            "Condense-AI/subnet-synthetic-dataset-v0.2",
            name="Conversations",
            split="train",
        )
        for qa_set in cached_qa_ds:
            item = QASet(**qa_set)
            self.redis.sadd(self.qa_key, item.model_dump_json())
        for conversation in cached_convo_ds:
            item = Conversation(**conversation)
            self.redis.sadd(self.convo_key, item.model_dump_json())

        logger.info(
            f"✅ Prefilled QA: {self.redis.scard(self.qa_key)} items. Conversations: {self.redis.scard(self.convo_key)} items."
        )

    def _get_next_context_seed(self):
        ds = random.choice(self.context_datasets)
        item = next(ds)
        return item["context"]

    def _get_next_messages_seed(self):
        ds = random.choice(self.instruct_datasets)
        item = next(ds)
        return item["messages"]

    async def _refresh_qa_queue(self):
        while self.running:
            if self.redis.scard(self.qa_key) < self.qa_config.max_items:
                try:
                    context_seed = self._get_next_context_seed()
                    (
                        questions,
                        answers,
                        total_chars,
                    ) = await self.generator.generate_qa_pairs(
                        context_seed, num_questions=self.qa_config.n_qa_per_context
                    )
                    qa_set = QASet(
                        questions=questions,
                        answers=answers,
                        total_chars=total_chars,
                        context_seed=context_seed,
                    )
                    self.redis.sadd(self.qa_key, qa_set.model_dump_json())
                    current_time = time.strftime("%H:%M:%S")
                    logger.info(
                        f"✅ QA Set: {self.redis.scard(self.qa_key)} - last_time: {current_time} - {total_chars} chars"
                    )
                except Exception as e:
                    logger.warning(f"❌ Error generating QA set: {e}")
            else:
                self.redis.spop(self.qa_key)
            await asyncio.sleep(self.refresh_time)

    async def _refresh_convo_queue(self):
        while self.running:
            if self.redis.scard(self.convo_key) < self.convo_config.max_items:
                try:
                    messages_seed = self._get_next_messages_seed()
                    messages, total_chars = await self.generator.generate_conversation(
                        messages_seed
                    )
                    conversation = Conversation(
                        messages=[Message(**msg) for msg in messages],
                        total_chars=total_chars,
                        messages_seed=[Message(**msg) for msg in messages_seed],
                    )
                    self.redis.sadd(self.convo_key, conversation.model_dump_json())
                    current_time = time.strftime("%H:%M:%S")
                    logger.info(
                        f"✅ Conversation: {self.redis.scard(self.convo_key)} - last_time: {current_time} - {total_chars} chars"
                    )
                except Exception as e:
                    logger.warning(f"❌ Error generating conversation: {e}")
            else:
                self.redis.spop(self.convo_key)
            await asyncio.sleep(self.refresh_time)

    def start(self):
        self.running = True
        self.loop.create_task(self._refresh_qa_queue())
        self.loop.create_task(self._refresh_convo_queue())

    def stop(self):
        self.running = False

    async def get_qas(self, n: int = 1) -> Optional[List[QASet]]:
        items = self.redis.srandmember(self.qa_key, n)
        return [QASet.model_validate_json(item) for item in items] if items else None

    async def get_conversations(self, n: int = 1) -> Optional[List[Conversation]]:
        items = self.redis.srandmember(self.convo_key, n)
        return (
            [Conversation.model_validate_json(item) for item in items]
            if items
            else None
        )

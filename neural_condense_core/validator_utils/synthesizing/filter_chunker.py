from datasets import load_dataset
from typing import Tuple
import random
from semantic_text_splitter import TextSplitter


class FilterExistanceChecker:
    def __init__(self, chunk_size: int = 256):
        self.splitter = TextSplitter(chunk_size)
        self.negative_dataset = self._load_negative_dataset()

    def _load_negative_dataset(self):
        negative_dataset = load_dataset(
            "TIGER-Lab/Fineweb-Instruct", streaming=True, split="train"
        )
        negative_dataset = negative_dataset.shuffle()
        negative_dataset = negative_dataset.filter(lambda x: len(x["response"]) > 1024)
        negative_dataset = negative_dataset.map(lambda x: {"text": x["response"]})
        negative_dataset = iter(negative_dataset)
        return negative_dataset

    def _get_negative_message(self):
        try:
            return next(self.negative_dataset)["text"]
        except StopIteration:
            self.negative_dataset = self._load_negative_dataset()
            return self._get_negative_message()

    def get_chunks(self, context: str) -> Tuple[str, str]:
        # Test on positive case (text from conversation)
        chunks = self.splitter.chunks(context)
        # Get random 2 chunks
        positive_chunks = random.sample(chunks, 2)
        # Test on negative case (text not from conversation)
        negative_chunks = [
            random.choice(self.splitter.chunks(self._get_negative_message()))
        ]
        return positive_chunks, negative_chunks

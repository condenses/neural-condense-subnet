from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List, Tuple
import random
from copy import deepcopy
from semantic_text_splitter import TextSplitter

class FilterExistanceChecker:
    def __init__(self):
        self.splitter = TextSplitter(256)
        self.negative_dataset = self._load_negative_dataset()

    def _load_negative_dataset(self):
        negative_dataset = load_dataset(
            "TIGER-Lab/Fineweb-Instruct", streaming=True, split="train"
        )
        negative_dataset = negative_dataset.shuffle()
        negative_dataset = negative_dataset.filter(lambda x: len(x["response"]) > 100)
        negative_dataset = negative_dataset.map(lambda x: {"text": x["response"]})
        negative_dataset = iter(negative_dataset)
        return negative_dataset

    def _get_negative_message(self):
        try:
            return next(self.negative_dataset)["text"]
        except StopIteration:
            self.negative_dataset = self._load_negative_dataset()
            return self._get_negative_message()

    def get_chunks(
        self, context: str
    ) -> Tuple[str, str]:
        # Test on positive case (text from conversation)
        chunks = self.splitter.chunks(context)
        positive_chunk = random.choice(chunks)
        # Test on negative case (text not from conversation)
        negative_chunk = random.choice(
            self.splitter.chunks(self._get_negative_message())
        )
        return positive_chunk, negative_chunk
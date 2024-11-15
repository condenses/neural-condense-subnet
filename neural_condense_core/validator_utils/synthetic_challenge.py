from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer
import re
import random
from typing import Iterator, List, Dict
import threading
from .custom_dataset_loaders import load_custom_dataset
from ..protocol import TextCompressProtocol


class Challenger:
    def __init__(self):
        """
        Initialize the Challenger class with various dataset loaders and configuration tokens.
        """
        self.raw_datasets: List[Iterator] = self._load_raw_dataset()
        self.qa_datasets: List[Iterator] = self._load_qa_dataset()
        self.conversation_datasets: List[Iterator] = self._load_conversation_dataset()
        self.sat_token = "[START-ACTIVATE-TOKEN]"
        self.eat_token = "[END-ACTIVATE-TOKEN]"
        self.lock = threading.Lock()  # Ensures thread safety for dataset access

    def __call__(
        self,
        tokenizer: AutoTokenizer,
        task: str = "",
        max_context_length_in_chars: int = 8000,
    ) -> TextCompressProtocol:
        """
        Generates a sample challenge based on the specified task type.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for formatting the text.
            task (str, optional): The type of task ('question_answering', 'conversation', 'reconstruction'). Defaults to "".
            max_context_length_in_chars (int, optional): Max length of context in characters. Defaults to 1536.

        Returns:
            TextCompressProtocol: An object containing context, activation prompt, and expected completion.
        """
        with self.lock:
            mixed_conversations, qa_question, qa_answer = (
                self._build_tasked_conversations(max_context_length_in_chars)
            )
            if task == "question_answering":
                return self._build_qa_task(
                    tokenizer, mixed_conversations, qa_question, qa_answer
                )
            elif task == "continual_conversation":
                return self._build_conversational_task(tokenizer, mixed_conversations)
            elif task == "reconstruction":
                return self._build_reconstruction_task(tokenizer, mixed_conversations)
            else:
                raise ValueError(f"Invalid task type: {task}")

    def _build_qa_task(
        self,
        tokenizer: AutoTokenizer,
        mixed_conversations: List[Dict[str, str]],
        qa_question: str,
        qa_answer: str,
    ) -> TextCompressProtocol:
        """
        Build the TextCompressProtocol for a question answering task.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for formatting.
            mixed_conversations (List[Dict[str, str]]): List of conversation messages.
            qa_question (str): Question to be answered.
            qa_answer (str): Answer to the question.

        Returns:
            TextCompressProtocol: The constructed protocol.
        """
        mixed_conversations[-1]["content"] += self.sat_token
        mixed_conversations.append(
            {"role": "user", "content": qa_question + self.eat_token}
        )
        mixed_conversations.append({"role": "assistant", "content": qa_answer})
        return self._build_protocol(tokenizer, mixed_conversations)

    def _build_conversational_task(
        self,
        tokenizer: AutoTokenizer,
        mixed_conversations: List[Dict[str, str]],
    ) -> TextCompressProtocol:
        """
        Get a conversation sample for the continual conversation task.
        """
        new_conversation = self._get_conversations(8096, num_conversation_pairs=1)
        mixed_conversations[-1]["content"] += self.sat_token
        new_conversation[0]["content"] += self.sat_token
        mixed_conversations.extend(new_conversation)
        return self._build_protocol(tokenizer, mixed_conversations)

    def _build_reconstruction_task(
        self,
        tokenizer: AutoTokenizer,
        mixed_conversations: List[Dict[str, str]],
    ) -> TextCompressProtocol:
        conversation_format = """
**[User]**: {user_message}

**[Assistant]**: {assistant_message}

---
(next conversation)
"""
        prompt = f"""
Please write above conversations in the following format:
{conversation_format}{self.eat_token}
"""
        expected_completion = ""
        for i in range(0, len(mixed_conversations), 2):
            expected_completion += conversation_format.format(
                user_message=mixed_conversations[i]["content"],
                assistant_message=mixed_conversations[i + 1]["content"],
            )
        mixed_conversations[-1]["content"] += self.sat_token

        mixed_conversations.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": expected_completion},
            ]
        )
        return self._build_protocol(tokenizer, mixed_conversations)

    def _build_tasked_conversations(
        self, max_context_length_in_chars: int
    ) -> List[Dict[str, str]]:
        conversations = self._get_conversations(max_context_length_in_chars // 4 * 3)
        qa_pairs = self._get_qa(max_context_length_in_chars // 4)
        qa_to_challenge = qa_pairs.pop()
        for qa_pair in qa_pairs:
            user_content = qa_pair["context"] + "\n" + qa_pair["question"]
            assistant_content = qa_pair["answer"]
            conversations.extend(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            )

        # Group by pairs of user and assistant messages then shuffle
        pair_wise_conversations = [
            conversations[i : i + 2] for i in range(0, len(conversations), 2)
        ]
        random.shuffle(pair_wise_conversations)
        # Flatten the list of pairs
        conversations = [
            item for sublist in pair_wise_conversations for item in sublist
        ]

        random_index_to_insert_qa = random.choice(range(0, len(conversations), 2))
        conversations[random_index_to_insert_qa]["content"] = (
            qa_to_challenge["context"]
            + "\n"
            + conversations[random_index_to_insert_qa]["content"]
        )
        total_conversations_length = sum(
            len(conversation["content"]) for conversation in conversations
        )
        remaining_length = max_context_length_in_chars - total_conversations_length
        if remaining_length > 128:
            raw_text = self._get_raw_text(max_context_length_in_chars)
            raw_text = raw_text[:remaining_length]
            raw_text = ".".join(raw_text.split(".")[:-1]) + "."
            # Select a random index to insert the raw text
            random_index_to_insert_raw = random.choice(range(0, len(conversations), 2))
            conversations[random_index_to_insert_raw]["content"] = (
                raw_text + "\n" + conversations[random_index_to_insert_raw]["content"]
            )

        return conversations, qa_to_challenge["question"], qa_to_challenge["answer"]

    def _get_conversations(
        self, max_context_length_in_chars: int, num_conversation_pairs: int = 10
    ) -> List[Dict[str, str]]:
        conversations = []
        current_length = 0
        for _ in range(10):
            convo_dataset = random.choice(self.conversation_datasets)
            for _ in range(5):
                try:
                    item = next(convo_dataset)
                    turn_length = len(item["user"]) + len(item["assistant"])
                    if current_length + turn_length <= max_context_length_in_chars:
                        current_length += turn_length
                        conversations.extend(
                            [
                                {"role": "user", "content": item["user"]},
                                {"role": "assistant", "content": item["assistant"]},
                            ]
                        )
                    else:
                        break
                except StopIteration:
                    break
            if len(conversations) / 2 >= num_conversation_pairs:
                break
        return conversations

    def _get_qa(self, max_context_length_in_chars: int) -> Dict[str, str]:
        current_length = 0
        qa_pairs = []
        for _ in range(5):
            qa_dataset = random.choice(self.qa_datasets)
            for _ in range(5):
                try:
                    item = next(qa_dataset)
                    context_length = len(item["context"])
                    question_length = len(item["question"])
                    if (
                        current_length + context_length + question_length
                        <= max_context_length_in_chars
                    ):
                        current_length += context_length + question_length
                        qa_pairs.append(item)
                    else:
                        break
                except StopIteration:
                    break
        return qa_pairs

    def _get_raw_text(self, max_context_length_in_chars: int) -> str:
        current_length = 0
        raw_text = ""
        for _ in range(5):
            raw_dataset = random.choice(self.raw_datasets)
            for _ in range(5):
                try:
                    item = next(raw_dataset)
                    text_length = len(item["text"])
                    if current_length + text_length <= max_context_length_in_chars:
                        current_length += text_length
                        raw_text += item["text"]
                    else:
                        break
                except StopIteration:
                    break
        return raw_text

    def _build_protocol(
        self,
        tokenizer: AutoTokenizer,
        messages: List[Dict[str, str]],
    ) -> TextCompressProtocol:
        """
        Build the TextCompressProtocol from the list of messages.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for formatting.
            messages (List[Dict[str, str]]): List of conversation messages.
            expected_completion (str, optional): Expected response text. Defaults to None.

        Returns:
            TextCompressProtocol: The constructed protocol.
        """
        messages_str = tokenizer.apply_chat_template(messages, tokenize=False)
        parts = re.split(
            f"{re.escape(self.sat_token)}|{re.escape(self.eat_token)}", messages_str
        )
        context, activation_prompt, expected_completion = parts

        return TextCompressProtocol(
            context=context.strip(),
            activation_prompt=activation_prompt.strip(),
            expected_completion=expected_completion.strip(),
        )

    def _load_qa_dataset(self) -> List[IterableDataset]:
        """
        Load and shuffle the QA datasets.

        Returns:
            List[IterableDataset]: Shuffled QA datasets.
        """
        datasets = [load_custom_dataset("squad_v2"), load_custom_dataset("coqa")]
        return [InfiniteDataset(ds.shuffle()) for ds in datasets]

    def _load_raw_dataset(self) -> List[IterableDataset]:
        """
        Load and shuffle raw text datasets.

        Returns:
            List[IterableDataset]: Shuffled raw text datasets.
        """
        datasets = [
            load_dataset("gair-prox/FineWeb-pro", streaming=True, split="train")
        ]
        return [InfiniteDataset(ds.shuffle()) for ds in datasets]

    def _load_conversation_dataset(self) -> List[IterableDataset]:
        """
        Load and shuffle conversation datasets.

        Returns:
            List[IterableDataset]: Shuffled conversation datasets.
        """
        datasets = [
            load_custom_dataset("infinity_instruct"),
            load_custom_dataset("open_math_instruct"),
        ]
        return [InfiniteDataset(ds.shuffle()) for ds in datasets]


class InfiniteDataset:
    def __init__(self, dataset: IterableDataset):
        """
        Initialize the InfiniteDataset wrapper.

        Args:
            dataset (IterableDataset): The dataset object to wrap. It should be an iterable dataset.
        """
        self.dataset = dataset
        self.iterator = iter(self.dataset)  # Initialize the iterator

    def __iter__(self) -> Iterator:
        """
        Return the iterator for the dataset.

        Returns:
            Iterator: An iterator over the dataset.
        """
        return self

    def __next__(self):
        """
        Get the next item in the dataset. Automatically reinitialize the iterator if the end is reached.

        Returns:
            The next item in the dataset.
        """
        try:
            return next(self.iterator)
        except StopIteration:
            # Reinitialize iterator if the end is reached
            self.iterator = iter(self.dataset)
            return next(self.iterator)  # Return the first item of the new iterator

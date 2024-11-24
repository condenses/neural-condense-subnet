from transformers import AutoTokenizer
import re
import substrateinterface as st
from .scheduler import Scheduler
from .convo_generator import ConvoGenerator
from .schemas import Message, QASchedulerConfig, ConversationSchedulerConfig
import random
import os
from typing import List, Tuple
from ...protocol import TextCompressProtocol
from ...constants import constants
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
            convo_config=ConversationSchedulerConfig(
                n_new_conversations=100, n_previous_conversations=2, max_items=100
            ),
            refresh_time=15.0,
        )
        self.synthesizer.start()
        self.start_activation_token = "<START-ACTIVATE-TOKEN>"
        self.end_activation_token = "<END-ACTIVATE-TOKEN>"
        self.task_to_builder = {
            "question_answering": self._build_qa_conversation,
            "causal_conversation": self._build_causal_conversation,
            "reconstruct_conversation": self._build_reconstruct_conversation,
            "trivial_qa_conversation": self._build_trivial_qa_conversation,
        }
        for task in constants.SYNTHETIC_TASK_CONFIG:
            assert (
                task.task in self.task_to_builder
            ), f"Task {task.task} not supported. Supported tasks: {list(self.task_to_builder.keys())}"

    @retry(max_attempts=3)
    async def generate_challenge(
        self,
        tokenizer: AutoTokenizer,
        task: str = "question_answering",
        max_context_length_in_chars: int = 10000,
    ) -> TextCompressProtocol:
        messages, hidden_messages = await self.task_to_builder[task](
            max_context_length_in_chars
        )
        return self._build_protocol(tokenizer, messages, hidden_messages)

    @retry(max_attempts=3)
    async def _build_trivial_qa_conversation(
        self, max_chars: int
    ) -> Tuple[List[Message], List[Message]]:
        messages, _ = await self._build_causal_conversation(max_chars)
        # Trivial question answering: Ask about fill in the blank sentences
        # Select a random sentence from the conversation and 2 nearby sentences
        content_sentences = [len(msg.content.split(".")) for msg in messages]
        # Get msg with most sentences
        selected_message_index = content_sentences.index(max(content_sentences))
        selected_message_content = messages[selected_message_index].content
        sentences = selected_message_content.split(".")
        sentence_index = random.randint(1, len(sentences) - 1)
        hidden_sentence = sentences[sentence_index]
        sentences[sentence_index] = "______"
        fill_in_the_blank_sentence = ".".join(
            sentences[max(sentence_index - 3, 0) : sentence_index + 3]
        )

        hidden_messages = [
            Message(
                role="user",
                content=f"From the conversation above, fill in the blank: {fill_in_the_blank_sentence}",
            ),
            Message(role="assistant", content=hidden_sentence),
        ]
        return messages, hidden_messages

    @retry(max_attempts=3)
    async def _build_reconstruct_conversation(
        self, max_chars: int
    ) -> Tuple[List[Message], List[Message]]:
        messages, _ = await self._build_causal_conversation(max_chars)

        turn_format = """[Role]: [Message]
Example:
- User: Hello, how are you?
- Assistant: I am fine, thank you.
- User: What is your name?
- Assistant: I am a helpful assistant.
... other turns

"""
        activation_prompt = f"Please paraphrase all the messages in the conversation above in the format {turn_format}"
        formatted_messages = [
            f"- {msg.role.capitalize()}: {msg.content}" for msg in messages
        ]
        hidden_messages = [
            Message(role="user", content=activation_prompt),
            Message(role="assistant", content="\n".join(formatted_messages)),
        ]
        return messages, hidden_messages

    @retry(max_attempts=3)
    async def _build_causal_conversation(
        self, max_chars: int
    ) -> Tuple[List[Message], List[Message]]:
        conversations = await self._ensemble_conversations(n=10)
        messages = []
        for conversation in conversations:
            messages.extend(conversation)

        # Find indext i that is the last index where the sum of the user and assistant messages is less than max_chars
        condense_chars = 0
        for i in range(0, len(messages) - 1, 2):
            user_content = messages[i].content
            assistant_content = messages[i + 1].content
            if condense_chars + len(user_content) + len(assistant_content) > max_chars:
                break
            condense_chars += len(user_content) + len(assistant_content)
        hidden_messages = messages[i:]
        return messages[:i], hidden_messages

    @retry(max_attempts=3)
    async def _build_qa_conversation(
        self, max_chars: int
    ) -> Tuple[List[Message], List[Message]]:
        main_qa_set = await self.synthesizer.get_qas(n=1)
        main_qa_set = main_qa_set[0]
        context = main_qa_set.context_seed
        qa_pairs = [
            (main_qa_set.questions[i], main_qa_set.answers[i])
            for i in range(len(main_qa_set.questions))
        ]
        random.shuffle(qa_pairs)
        selected_question, selected_answer = qa_pairs.pop()

        hidden_messages: List[Message] = []
        for q, a in qa_pairs:
            hidden_messages.extend(
                [
                    Message(role="user", content=q),
                    Message(role="assistant", content=a),
                ]
            )

        qa_seed = [
            Message(role="user", content=f"{context}\n{selected_question}"),
            Message(role="assistant", content=selected_answer),
        ]

        conversations = await self._ensemble_conversations(10)
        if not conversations:
            return qa_seed, hidden_messages

        messages: List[Message] = []
        remaining_chars = (
            max_chars - len(context) - len(selected_question) - len(selected_answer)
        )

        while remaining_chars > 0 and conversations:
            conversation = conversations.pop()
            for i in range(0, len(conversation) - 1, 2):
                user_message = conversation[i]
                assistant_message = conversation[i + 1]
                if remaining_chars <= len(user_message.content) + len(
                    assistant_message.content
                ):
                    break
                messages.extend([user_message, assistant_message])
                remaining_chars -= len(user_message.content) + len(
                    assistant_message.content
                )

        messages.extend(qa_seed)
        return messages, hidden_messages

    def _build_protocol(
        self,
        tokenizer: AutoTokenizer,
        messages: List[Message],
        hidden_messages: List[Message],
    ) -> TextCompressProtocol:
        messages[-1].content = messages[-1].content + self.start_activation_token
        hidden_messages[0].content = (
            hidden_messages[0].content + self.end_activation_token
        )

        all_messages = [msg.model_dump() for msg in messages + hidden_messages]
        prompt = tokenizer.apply_chat_template(
            all_messages, tokenize=False, add_generation_prompt=False
        )

        context, activation_prompt, expected_completion = re.split(
            f"{re.escape(self.start_activation_token)}|{re.escape(self.end_activation_token)}",
            prompt,
        )
        return TextCompressProtocol(
            context=context,
            activation_prompt=activation_prompt,
            expected_completion=expected_completion,
        )

    async def _ensemble_conversations(self, n: int) -> List[List[Message]]:
        qa_conversations = await self._get_qa_as_conversation(n=n)
        causal_conversations = await self.synthesizer.get_conversations(n=n)
        if not causal_conversations:
            return qa_conversations

        all_conversations = qa_conversations + [
            conversation.messages for conversation in causal_conversations
        ]
        random.shuffle(all_conversations)
        return all_conversations

    async def _get_qa_as_conversation(self, n: int) -> List[List[Message]]:
        multiple_conversations: List[List[Message]] = []
        qa_sets = await self.synthesizer.get_qas(n=n)
        if not qa_sets:
            return []

        for qa_set in qa_sets:
            conversation: List[Message] = []
            context = qa_set.context_seed
            qa_pairs = [
                (qa_set.questions[i], qa_set.answers[i])
                for i in range(len(qa_set.questions))
            ]
            random.shuffle(qa_pairs)
            for q, a in qa_pairs:
                if not conversation:
                    conversation.append(Message(role="user", content=f"{context}\n{q}"))
                    conversation.append(Message(role="assistant", content=a))
                else:
                    conversation.append(Message(role="user", content=q))
                    conversation.append(Message(role="assistant", content=a))
            multiple_conversations.append(conversation)
        return multiple_conversations

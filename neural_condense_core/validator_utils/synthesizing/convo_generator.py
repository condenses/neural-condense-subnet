from typing import Dict, List, Optional
import httpx
import substrateinterface as st
import time
import random


class ConvoGenerator:
    def __init__(
        self,
        keypair: st.Keypair,
    ):
        self.model_id = "chat-llama-3-1-8b"
        self.model_ids = [
            "chat-llama-3-1-8b",
            "chat-llama-3-1-70b",
            "chat-llama-3-2-3b",
        ]
        self.url = "https://api.nineteen.ai/v1/chat/completions"
        self.keypair = keypair
        self.client = httpx.AsyncClient()

    def _get_headers(self):
        nonce = str(time.time_ns())
        signature = f"0x{self.keypair.sign(nonce).hex()}"
        return {
            "validator-hotkey": self.keypair.ss58_address,
            "signature": signature,
            "nonce": nonce,
            "netuid": "47",
            "Content-Type": "application/json",
        }

    def _get_assistant_messages(self, messages, n_few_shots):
        a_messages = messages[n_few_shots:]  # Skip few shots
        for i in range(len(a_messages)):
            if a_messages[i]["role"] == "assistant":
                a_messages[i]["role"] = "user"
            else:
                a_messages[i]["role"] = "assistant"
        return a_messages

    async def _make_api_call(self, messages, sampling_params):
        payload = sampling_params | {
            "model": random.choice(self.model_ids),
            "messages": messages,
        }
        response = await self.client.post(
            self.url, json=payload, headers=self._get_headers(), timeout=32
        )
        if response.status_code != 200:
            raise Exception(f"Nineteen API Error: {response.text}")
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content

    async def generate_conversation(
        self,
        messages_seed: Optional[List[Dict[str, str]]] = None,
        max_turns: int = 4,
    ):
        assert (
            messages_seed[0]["role"] == "user"
            and messages_seed[-1]["role"] == "assistant"
        ), "First and last message must be a user and assistant message respectively"
        assert (
            len(messages_seed) % 2 == 0
        ), "messages_seed must have an even number of messages"

        reversed_messages_seed = []
        role_switcher = {"user": "assistant", "assistant": "user"}
        for message in messages_seed:
            content = message["content"]
            role = message["role"]
            reversed_messages_seed.append(
                {"role": role_switcher[role], "content": content}
            )
        assert max_turns % 2 == 0, "max_turns must be even"
        messages = [
            {
                "role": "user",
                "content": (
                    "Your task is to act as a human and questioning on me. "
                    "You can ask me anything, I will give you the answer."
                    "You have to talk like a human, concisely and dont show emotion."
                ),
            },
            {
                "role": "assistant",
                "content": "Sure, we will start with a simple question: 1+1=?",
            },
            {
                "role": "user",
                "content": "2",
            },
        ]
        n_few_shots = len(messages)
        messages.extend(reversed_messages_seed)
        sampling_params = {"temperature": 0.4, "max_tokens": 1024, "stream": False}
        # Get first response
        text = await self._make_api_call(messages, sampling_params)
        messages.append({"role": "assistant", "content": text})

        # Generate multiple conversation turns
        assistant_messages = self._get_assistant_messages(messages, n_few_shots)
        for i in range(max_turns):
            # CALL ASSISTANT-MESSAGES -> ASSISTANT-MESSAGES
            text = await self._make_api_call(assistant_messages, sampling_params)
            assistant_messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": text})
            if i == max_turns - 1:
                break
            # CALL MESSAGES -> FAKE--MESSAGES
            text = await self._make_api_call(messages, sampling_params)
            assistant_messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": text})

        total_chars = 0
        for i in range(len(assistant_messages)):
            total_chars += len(assistant_messages[i]["content"])
        return assistant_messages, total_chars

    async def generate_qa_pairs(self, context_seed: str, num_questions: int = 1):
        question_description = "The question can vary in complexity, ranging from simple tasks like extracting entities or events to more nuanced queries requiring analysis, summarization, or synthesis based on the given context"
        prompt = f"- Context:\n---\n{context_seed}\n---\n Generate {num_questions} different questions."
        system_message = (
            "Your task is to understand the provided context and generate questions about it."
            f"{question_description}"
            "Each question should be in the following format: - Question: <question>?\n"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        sampling_params = {
            "temperature": 1.0,
            "stop": ["?"],
            "max_tokens": 512,
            "stream": False,
        }
        text = await self._make_api_call(messages, sampling_params)
        questions = self._extract_questions(text)
        if not questions:
            print(text)
        answers = []
        for question in questions:
            sampling_params = {"temperature": 0.4, "max_tokens": 1024, "stream": False}
            text = await self._make_api_call(
                [{"role": "user", "content": f"{context_seed}\n\n{question}"}],
                sampling_params,
            )
            answers.append(text)
        total_chars = len(context_seed)
        for q, a in zip(questions, answers):
            total_chars += len(q) + len(a)
        return questions, answers, total_chars

    def _extract_questions(self, text: str, prefix: str = "Question: "):
        lines = text.split("\n")
        questions = []
        for line in lines:
            if prefix in line:
                question = line[line.index(prefix) + len(prefix) :].strip()
                if len(question) > 16:
                    questions.append(question)
        return questions

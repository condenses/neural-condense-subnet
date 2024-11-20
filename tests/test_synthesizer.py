from transformers import AutoTokenizer
from neural_condense_core.validator_utils.synthesizing import ChallengeGenerator
import json
import os

os.makedirs("tmp", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-7B-Instruct-v0.2")

tasks = [
    "question_answering",
    "causal_conversation",
    "reconstruct_conversation",
    "trivial_qa_conversation",
]

challenge_generator = ChallengeGenerator()

for task in tasks:
    challenge = challenge_generator.generate_challenge(tokenizer, task=task)
    json.dump(challenge.deserialize(), open(f"tmp/challenge_{task}.json", "w"))

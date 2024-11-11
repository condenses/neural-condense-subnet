from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import torch.nn.functional as F
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)
from .datatypes import BatchedScoringRequest
from .utils import loss_to_scores, base64_to_ndarray, _smooth_scores, unit_test
import threading
import traceback
import random
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Validator-Backend")


class ScoringService:
    def __init__(self):
        """
        Initializes the ScoringService with model and tokenizer storage, device configuration,
        and a lock for thread-safe operations. Runs a unit test to verify setup.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.lock = threading.Lock()
        unit_test(self)

    def load_model(self, model_name: str):
        """
        Loads a specified model and tokenizer if not already loaded, ensuring thread safety.
        """
        with self.lock:
            try:
                if model_name not in self.models:
                    self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_name
                    )
                    self.models[model_name].to(self.device)
                    self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                        model_name
                    )
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

    @torch.no_grad()
    def get_scoring(self, request: BatchedScoringRequest):
        """
        Returns scoring based on criteria specified in the request, such as loss and accuracy,
        calculated from the given model and tokenizer.
        """
        try:
            model_name = request.ground_truth_request.model_name
            self.load_model(model_name)
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            logs = {}
            criteria = random.choice(request.ground_truth_request.criterias)
            for miner_response in request.miner_responses:
                miner_response.compressed_tokens = base64_to_ndarray(
                    miner_response.compressed_tokens_b64
                )

            if criteria == "loss":
                negative_losses = self.calculate_loss_criteria(
                    request, model, tokenizer
                )
                logs["losses"] = negative_losses
                scores = _smooth_scores(negative_losses, delta_0=0.4, decay=0.7)

            if criteria == "accuracy":
                scores = self.calculate_accuracy_criteria(request, model, tokenizer)
                logs["accuracy"] = scores

            return {"scores": scores, "logs": logs}

        except Exception as e:
            traceback.print_exc()
            print(f"Error in get_scoring: {e}")
            return {"scores": []}

    def calculate_loss_criteria(self, request: BatchedScoringRequest, model, tokenizer):
        """
        Calculates the loss-based scores by comparing expected and generated tokens based
        on the activation prompt and expected completion.
        """
        device = model.device
        activation_prompt = request.ground_truth_request.activation_prompt
        expected_completion = request.ground_truth_request.expected_completion
        losses = []

        try:
            activation_prompt_tokens = tokenizer(
                activation_prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            expected_completion_tokens = tokenizer(
                expected_completion, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            activation_prompt_embeddings = model.get_input_embeddings()(
                activation_prompt_tokens
            )
            expected_completion_embeddings = model.get_input_embeddings()(
                expected_completion_tokens
            )

            for miner_response in request.miner_responses:
                try:
                    compressed_tokens = (
                        torch.tensor(miner_response.compressed_tokens)
                        .unsqueeze(0)
                        .to(device)
                    )
                    inputs_embeddings = torch.cat(
                        [
                            compressed_tokens,
                            activation_prompt_embeddings,
                            expected_completion_embeddings,
                        ],
                        dim=1,
                    ).to(device)

                    labels = torch.cat(
                        [
                            torch.full(
                                (1, compressed_tokens.shape[1]), -100, dtype=torch.long
                            ).to(device),
                            activation_prompt_tokens,
                            expected_completion_tokens,
                        ],
                        dim=1,
                    ).to(device)

                    labels = labels[:, 1:]
                    outputs = model(inputs_embeds=inputs_embeddings)
                    logits = outputs.logits[:, :-1, :]

                    loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                    losses.append(loss.item())
                except Exception as e:
                    print(f"Error in calculate_loss_criteria loop: {e}")
                    losses.append(1000)
            scores = loss_to_scores(losses)
            logger.info(f"Losses: {losses}")
            return scores
        except Exception as e:
            print(f"Error in calculate_loss_criteria: {e}")
            return []

    def calculate_accuracy_criteria(
        self, request: BatchedScoringRequest, model, tokenizer
    ):
        """
        Calculates accuracy-based scores by generating responses from miner responses,
        comparing them to the expected completion.
        """
        device = model.device
        activation_prompt = request.ground_truth_request.activation_prompt
        expected_completion = request.ground_truth_request.expected_completion
        accuracy_scores = []

        try:
            activation_prompt_tokens = tokenizer(
                activation_prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            activation_prompt_embeddings = model.get_input_embeddings()(
                activation_prompt_tokens
            )

            for miner_output in request.miner_responses:
                try:
                    compressed_tokens = (
                        torch.tensor(miner_output.compressed_tokens)
                        .unsqueeze(0)
                        .to(device)
                    )
                    inputs_embeds = torch.cat(
                        [compressed_tokens, activation_prompt_embeddings], dim=1
                    ).to(device)

                    generated_outputs = model.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=64,
                        num_return_sequences=1,
                    )
                    completion = (
                        tokenizer.decode(
                            generated_outputs[0], skip_special_tokens=True
                        ).strip()
                        or "I dont know"
                    )
                    accuracy = self._llm_judge(
                        expected_completion, completion, model, tokenizer
                    )
                    accuracy_scores.append(accuracy)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error in calculate_accuracy_criteria loop: {e}")
                    accuracy_scores.append(0)
            return accuracy_scores
        except Exception as e:
            traceback.print_exc()
            print(f"Error in calculate_accuracy_criteria: {e}")
            return []

    def _llm_judge(
        self, expected_completion, completion, model, tokenizer, max_new_tokens=32
    ):
        """
        Generates a yes or no judgment on the accuracy of the model's completion compared to
        the expected completion.
        """
        try:
            prompt = f"""Task description: Given a ground truth completion and a model completion, answer concisely: yes if the model completion is correct, and no otherwise. 
            - Ground truth completion: {expected_completion}
            - Model completion: {completion}
            """
            # Remove special tokens and instruction tags, TODO: make it general
            prompt = prompt.replace("</s>", "").replace("[/INST]", "")
            pipeline = TextGenerationPipeline(model, tokenizer, device=self.device)
            messages = [{"role": "user", "content": prompt}]
            completion_text = pipeline(
                messages,
                return_full_text=False,
                max_new_tokens=max_new_tokens,
            )[0]["generated_text"]
            logger.info(
                (
                    f"Expected: {expected_completion}\n"
                    f"Completion: {completion}\n"
                    f"Prompt: {prompt}\n"
                    f"Generated: {completion_text}"
                )
            )
            return "yes" in completion_text
        except Exception as e:
            traceback.print_exc()
            print(f"Error in _llm_judge: {e}")
            return True


app = FastAPI()
scoring_service = ScoringService()


@app.get("/")
def is_alive():
    """
    Endpoint to check if the service is running and responsive.
    """
    return {"message": "I'm alive!"}


@app.post("/scoring")
def get_scoring(request: BatchedScoringRequest):
    """
    Endpoint to receive a batched scoring request and return calculated scores.
    """
    try:
        return scoring_service.get_scoring(request)
    except Exception as e:
        print(f"Error in /scoring endpoint: {e}")
        return {"error": "Failed to calculate scores"}

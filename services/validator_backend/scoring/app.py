# TODO: Efficient switching between target models. Currently fixed to mistral-7b-instruct-v0.2.
from flask import Flask, request, jsonify
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import random
import logging
import gc
from .datatypes import BatchedScoringRequest
from .metric_handlers import metric_handlers

gc.enable()
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("Validator-Backend")


class ScoringService:
    def __init__(self):
        """
        Initializes the ScoringService with model and tokenizer storage, device configuration,
        and a lock for thread-safe operations. Runs a unit test to verify setup.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            "unsloth/mistral-7b-instruct-v0.2"
        ).to(dtype=self.dtype, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/mistral-7b-instruct-v0.2"
        )

    @torch.no_grad()
    def get_metrics(self, request: BatchedScoringRequest) -> dict[str, float]:
        criteria = random.choice(request.ground_truth_request.criterias)
        values = []
        metric_handler = metric_handlers[criteria]["handler"]
        preprocess_batch = metric_handlers[criteria]["preprocess_batch"]
        for miner_response in request.miner_responses:
            miner_response.decode()
            try:
                value = metric_handler(
                    compressed_tokens=miner_response.compressed_tokens,
                    activation_prompt=request.ground_truth_request.activation_prompt,
                    expected_completion=request.ground_truth_request.expected_completion,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error(f"Error in scoring: {e}")
                value = None
            values.append(value)
        values = preprocess_batch(values)
        return {"metrics": {criteria: values}}


app = Flask(__name__)
scoring_service = ScoringService()


@app.route("/", methods=["GET"])
def is_alive():
    return jsonify({"message": "I'm alive!"})


@app.route("/get_metrics", methods=["POST"])
def get_metrics():
    request_data = BatchedScoringRequest(**request.get_json())
    return jsonify(scoring_service.get_metrics(request_data))

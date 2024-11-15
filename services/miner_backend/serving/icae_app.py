import torch
from peft import LoraConfig
from .ICAE import ICAE, ModelArguments
from safetensors.torch import load_file
import rich
import numpy as np
import base64
import io
import time
from flask import Flask, request, jsonify

app = Flask(__name__)


def ndarray_to_base64(array: np.ndarray) -> str:
    """Convert a NumPy array to a base64-encoded string."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    return base64_str


class InferenceLogger:
    """
    Logger class for inference processes. This logs key-value pairs of information
    during the execution of the backend inference, using the rich library for better readability.
    """

    @staticmethod
    def log(key, value):
        rich.print(f"Inference Backend -- {key}: {value}")


class Inference:
    def __init__(self, model_args, lora_config):
        self.model_args = model_args
        self.lora_config = lora_config
        self.device = None
        self.model_compress = None

    def setup(self, device):
        self.device = device
        self.model_compress = ICAE(self.model_args, self.lora_config)
        state_dict = load_file(self.model_args.checkpoint_path)
        self.model_compress.load_state_dict(
            state_dict, strict=False
        )  # only load lora and memory token embeddings
        self.model_compress = self.model_compress.to(self.device)

    @torch.no_grad()
    def predict(self, context, target_model):
        """
        Performs the prediction by compressing the input context and logging the compression details.

        Args:
            context (str): The input text context to compress.
            target_model (str): The target model name.

        Returns:
            dict: Compressed context and the target model name for further processing.
        """
        t1 = time.time()
        tokenized_input = self.model_compress.tokenizer(
            context,
            truncation=True,
            max_length=5120,
            padding=False,
            return_attention_mask=False,
        )
        input_ids = torch.LongTensor([tokenized_input["input_ids"]]).to(self.device)
        memory_slots = self.model_compress._compress(input_ids)

        InferenceLogger.log(
            "Predict",
            f"Compress token length {len(memory_slots)} shape{memory_slots.shape}",
        )
        InferenceLogger.log("Inference time", time.time() - t1)

        memory_slots = memory_slots.cpu().numpy()
        memory_slots_b64 = ndarray_to_base64(memory_slots)
        return {
            "compressed_tokens_b64": memory_slots_b64,
            "target_model": target_model,
        }


# Initialize inference outside the route to avoid re-initialization with every request
model_args = ModelArguments()
lora_config = LoraConfig(
    r=512,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)

inference = Inference(model_args=model_args, lora_config=lora_config)
inference.setup(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


@app.route("/condense", methods=["POST"])
def predict():
    """
    Endpoint for prediction requests. Receives JSON data with 'context' and 'target_model'.
    """
    data = request.get_json()
    context = data.get("context")
    target_model = data.get("target_model")

    if not context or not target_model:
        return (
            jsonify({"error": "Missing 'context' or 'target_model' in request."}),
            400,
        )

    prediction = inference.predict(context, target_model)
    return jsonify(prediction)

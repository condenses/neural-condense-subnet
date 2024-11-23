import base64
import io
from flask import Flask, request, jsonify
import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress import KnormPress
import torch
from .utils import upload_to_minio
import os
import minio


class KVPressService:
    def __init__(self):
        self.device = "cuda:0"
        self.ckpt = "Condense-AI/Mistral-7B-Instruct-v0.2"
        self.bucket_name = os.getenv("MINIO_BUCKET", "condense_miner")
        # Initialize model components
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(self.ckpt).to(self.device)
        self.press = KnormPress(compression_ratio=0.75)

        # Initialize MinIO client
        self.minio_client = minio.Minio(
            os.getenv("MINIO_SERVER", "minio.condenses.ai").split("://")[1],
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=True,
        )

        # Validate model setup
        self._validate_model_setup()

    def _validate_model_setup(self):
        """Validate the model setup with dummy inputs"""
        inputs = self.model.dummy_inputs["input_ids"].to(self.device)

        with torch.no_grad():
            original_shape = self.model(inputs).past_key_values[0][0].shape

        with torch.no_grad(), self.press(self.model):
            compressed_shape = self.model(inputs).past_key_values[0][0].shape

        print(f"Original shape: {original_shape}")
        print(f"Compressed shape: {compressed_shape}")

    def compress_context(self, context: str) -> tuple[str, str]:
        """Compress context and store KV pairs"""
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(
            self.device
        )

        with torch.no_grad(), self.press(self.model):
            past_key_values = self.model(input_ids).past_key_values

        DynamicCache(past_key_values)

        # Convert to numpy arrays
        numpy_past_key_values = tuple(
            tuple(tensor.cpu().numpy() for tensor in tensors)
            for tensors in past_key_values
        )
        print(numpy_past_key_values[0][0].shape)

        # Generate unique filename using timestamp
        filename = f"{int(time.time_ns())}.npy"

        # Upload to MinIO
        upload_to_minio(
            self.minio_client, self.bucket_name, filename, numpy_past_key_values
        )

        return f"{self.minio_client.endpoint_url}/{self.bucket_name}/{filename}"


# Initialize Flask app and KVPress service
app = Flask(__name__)
kv_service = KVPressService()


@app.route("/condense", methods=["POST"])
def compress_endpoint():
    """Endpoint for compressing context"""
    data = request.get_json()
    context = data.get("context")
    target_model = data.get("target_model")

    if not context:
        return jsonify({"error": "Missing 'context' in request."}), 400

    try:
        compressed_kv_url = kv_service.compress_context(context)
        return jsonify(
            {"target_model": target_model, "compressed_kv_url": compressed_kv_url}
        )
    except Exception as e:
        return jsonify({"error": "Failed to process request", "details": str(e)}), 500

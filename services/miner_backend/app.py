from flask import Flask, request, jsonify
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress import KnormPress
import torch
from .soft_token.soft_token_condenser_modeling import Condenser
import os
import minio
import structlog
from .utils import upload_to_minio
import argparse

logger = structlog.get_logger()


class CompressionService:
    def __init__(self, algorithm: str):
        self.dtype = torch.bfloat16
        self.algorithm = algorithm
        self._init_minio_client()
        self._init_model()

    def _init_minio_client(self):
        """Initialize MinIO client and validate config"""
        self.bucket_name = os.getenv("MINIO_BUCKET", "condense_miner")
        self.endpoint_url = os.getenv("MINIO_SERVER")

        self._validate_minio_config()

        self.minio_client = minio.Minio(
            self.endpoint_url.replace("http://", "").replace("https://", ""),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=False,
        )

    def _init_model(self):
        """Initialize model based on selected algorithm"""
        self.device = "cuda"

        if self.algorithm == "kvpress":
            self.ckpt = "Condense-AI/Mistral-7B-Instruct-v0.2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.ckpt, torch_dtype=self.dtype
            ).to(self.device)
            self.press = KnormPress(compression_ratio=0.75)

        elif self.algorithm == "soft_token":
            self.ckpt = "Condense-AI/Mistral-7B-Instruct-v0.2"
            self.repo_id = "Condense-AI/Soft-Token-Condenser-Llama-3.2-1B"
            self.condenser = Condenser.from_pretrained(self.repo_id, dtype=self.dtype)
            self.condenser.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.ckpt, torch_dtype=self.dtype
            ).to(self.device)
            self.press = KnormPress(compression_ratio=0.75)

        elif self.algorithm == "activation_beacon":
            self.ckpt = "namespace-Pt/ultragist-mistral-7b-inst"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.ckpt,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.ckpt,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                attn_implementation="sdpa",
                ultragist_ratio=[4],
            ).to(self.device)

    @torch.no_grad()
    def compress_context(self, context: str) -> str:
        """Compress context using selected algorithm"""
        if self.algorithm == "kvpress":
            return self._compress_kvpress(context)
        elif self.algorithm == "soft_token":
            return self._compress_soft_token(context)
        elif self.algorithm == "activation_beacon":
            return self._compress_activation_beacon(context)

    def _compress_kvpress(self, context: str) -> str:
        input_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids.to(
            self.device
        )

        with torch.no_grad(), self.press(self.model):
            past_key_values = self.model(input_ids, num_logits_to_keep=1).past_key_values

        return self._save_and_return_url(past_key_values)

    def _compress_soft_token(self, context: str) -> str:
        compressed_tokens = self.condenser.compress(context)

        with torch.no_grad(), self.press(self.model):
            past_key_values = self.model(
                inputs_embeds=compressed_tokens
            ).past_key_values

        return self._save_and_return_url(past_key_values)

    def _compress_activation_beacon(self, context: str) -> str:
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(
            self.device
        )

        self.model.memory.reset()
        self.model(input_ids=input_ids)
        past_key_values = self.model.memory.get_memory()

        # Log metrics specific to activation beacon
        ultragist_size, raw_size, sink_size = self.model.memory.get_memory_size()
        logger.info(
            "model_metrics",
            ultragist_size=ultragist_size,
            raw_size=raw_size,
            sink_size=sink_size,
        )

        return self._save_and_return_url(past_key_values)

    def _save_and_return_url(self, past_key_values):
        """Process output and save to MinIO"""
        DynamicCache(past_key_values)

        numpy_past_key_values = tuple(
            tuple(tensor.to(dtype=torch.float32).cpu().numpy() for tensor in tensors)
            for tensors in past_key_values
        )

        filename = f"{int(time.time_ns())}.npy"
        upload_to_minio(
            self.minio_client, self.bucket_name, filename, numpy_past_key_values
        )

        return f"{self.endpoint_url}/{self.bucket_name}/{filename}"

    def _validate_minio_config(self):
        """Validate MinIO configuration"""
        if not self.endpoint_url:
            raise ValueError("MINIO_SERVER is not set")
        if not self.bucket_name:
            raise ValueError("MINIO_BUCKET is not set")
        if not os.getenv("MINIO_ACCESS_KEY"):
            raise ValueError("MINIO_ACCESS_KEY is not set")
        if not os.getenv("MINIO_SECRET_KEY"):
            raise ValueError("MINIO_SECRET_KEY is not set")
        if not (
            self.endpoint_url.startswith("http://")
            or self.endpoint_url.startswith("https://")
        ):
            raise ValueError("MINIO_SERVER must start with http:// or https://")


def create_app(algorithm):
    app = Flask(__name__)
    service = CompressionService(algorithm)

    @app.route("/condense", methods=["POST"])
    def compress_endpoint():
        """Endpoint for compressing context"""
        data = request.get_json()
        context = data.get("context")
        target_model = data.get("target_model")

        if not context:
            return jsonify({"error": "Missing 'context' in request"}), 400

        try:
            compressed_kv_url = service.compress_context(context)
            return jsonify(
                {"target_model": target_model, "compressed_kv_url": compressed_kv_url}
            )
        except Exception as e:
            logger.exception("compression_failed", error=str(e))
            return (
                jsonify({"error": "Failed to process request", "details": str(e)}),
                500,
            )

    return app


# This allows direct running of the file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        default="kvpress",
        choices=["kvpress", "soft_token", "activation_beacon"],
    )
    args = parser.parse_args()
    app = create_app(args.algorithm)
    app.run()

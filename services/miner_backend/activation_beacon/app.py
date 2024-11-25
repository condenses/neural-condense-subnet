from flask import Flask, request, jsonify
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
from .utils import upload_to_minio
import os
import minio


class ABCompressor:
    def __init__(self):
        self.device = "cuda"
        self.ckpt = "namespace-Pt/ultragist-mistral-7b-inst"
        self.bucket_name = os.getenv("MINIO_BUCKET", "condense_miner")
        # Initialize model components
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.ckpt,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            # you can manually set the compression ratio, otherwise the model will automatically choose the most suitable compression ratio from [2,4,8,16,32]
            ultragist_ratio=[4],
        ).to(self.device)

        # Initialize MinIO client
        self.minio_client = minio.Minio(
            os.getenv("MINIO_SERVER", "minio.condenses.ai").split("://")[1],
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=True,
        )

    def compress_context(self, context: str) -> tuple[str, str]:
        """Compress context and store KV pairs"""
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(
            self.device
        )

        self.model.memory.reset()
        self.model(input_ids=input_ids)
        past_key_values = self.model.memory.get_memory()
        ultragist_size, raw_size, sink_size = self.model.memory.get_memory_size()
        print(f"UltraGist size:   {ultragist_size}")
        print(f"Raw size:         {raw_size}")
        print(f"Sink size:        {sink_size}")
        print(f"Memory:           {past_key_values[0][0].shape}")
        print("*" * 20)
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
kv_service = ABCompressor()


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

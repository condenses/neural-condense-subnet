from flask import Flask, request, jsonify
from llmlingua import PromptCompressor
import structlog
import argparse

logger = structlog.get_logger()
logger.info("This will show in Universal Miner Backend logs")


class CompressionService:
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self._init_compressor()

    def _init_compressor(self):
        """Initialize model based on selected algorithm"""
        self.device = "cuda"

        if self.algorithm == "llmlingua":
            self.compressor = PromptCompressor()
        elif self.algorithm == "llmlingua-2":
            self.compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
            )

    def compress_context(self, context: str) -> str:
        """Compress context using selected algorithm"""
        if self.algorithm == "llmlingua":
            compressed_prompt = self.compressor.compress_prompt(
                context, instruction="", question="", target_token=200
            )
        elif self.algorithm == "llmlingua-2":
            compressed_prompt = self.compressor.compress_prompt(
                context, rate=0.7, force_tokens=["\n", "?"]
            )
        return compressed_prompt["compressed_prompt"]


def create_app(algorithm):
    app = Flask(__name__)
    service = CompressionService(algorithm)

    @app.route("/condense", methods=["POST"])
    def compress_endpoint():
        logger.info("Join compression endpoint")
        """Endpoint for compressing context"""
        data = request.get_json()
        context = data.get("context")
        target_model = data.get("target_model")

        if not context:
            return jsonify({"error": "Missing 'context' in request"}), 400

        try:
            compressed_context = service.compress_context(context)
            return jsonify(
                {"target_model": target_model, "compressed_context": compressed_context}
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
        default="llmlingua-2",
        choices=["llmlingua", "llmlingua-2"],
    )
    args = parser.parse_args()
    app = create_app(args.algorithm)
    app.run()

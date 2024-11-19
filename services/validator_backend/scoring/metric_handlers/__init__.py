from .perplexity import perplexity, preprocess_batch as perplexity_preprocess_batch

metric_handlers = {
    "perplexity": {
        "handler": perplexity,
        "preprocess_batch": perplexity_preprocess_batch,
    },
}

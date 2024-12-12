from .accuracy import accuracy, preprocess_batch as accuracy_preprocess_batch

metric_handlers = {
    "accuracy": {
        "handler": accuracy,
        "preprocess_batch": accuracy_preprocess_batch,
    },
}

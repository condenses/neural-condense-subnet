import rich


class InferenceLogger:
    """
    Logger class for inference processes. This logs key-value pairs of information
    during the execution of the backend inference, using the rich library for better readability.
    """

    @staticmethod
    def log(key, value):
        rich.print(f"Inference Backend -- {key}: {value}")

import time
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from neural_condense_core import validator_utils


def benchmark_challenger(
    n_iterations: int = 5000,
    max_characters: int = 10000,
    model_name: str = "Condense-AI/Mistral-7B-Instruct-v0.2",
):
    """
    Benchmark the Challenger model's response times and dataset creation for various tasks.

    Args:
        n_iterations (int): Number of iterations per task to perform. Defaults to 5000.
        max_characters (int): Maximum character limit for context in each task. Defaults to 10000.
        model_name (str): The name of the model to use for tokenization. Defaults to "Condense-AI/Mistral-7B-Instruct-v0.2".

    Returns:
        dict: Summary of benchmark results including average time per task and statistics on context length.
    """
    # Load tokenizer and initialize Challenger instance
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    challenger = validator_utils.Challenger()

    # Define task types and initialize logs
    tasks = ["question_answering", "reconstruction", "continual_conversation"]
    time_logs = {task: 0 for task in tasks}
    error_count = 0
    dataset_items = []
    context_lengths = []

    # Start progress bar for total iterations
    total_iterations = n_iterations * len(tasks)
    pbar = tqdm(total=total_iterations, desc="Benchmarking", unit="task")

    for i in range(n_iterations):
        for task in tasks:
            try:
                start_time = time.time()

                # Generate protocol using Challenger
                protocol = challenger(tokenizer, task, max_characters)

                # Record details of the generated sample
                item = {
                    "task": task,
                    "id": i,
                    "context": protocol.context,
                    "activation_prompt": protocol.activation_prompt,
                    "expected_completion": protocol.expected_completion,
                    "model_id": model_name,
                    "max_characters": max_characters,
                }

                # Track time taken for task
                time_logs[task] += time.time() - start_time

                # Store context length for analysis
                context_lengths.append(len(item["context"]))

                # Add item to dataset items
                dataset_items.append(item)

            except Exception as e:
                print(f"Error during task '{task}' at iteration {i}: {e}")
                error_count += 1
                continue

            # Update progress bar
            pbar.update(1)

    # Close progress bar
    pbar.close()

    # Calculate average processing time per task
    avg_time_logs = {
        task: total_time / n_iterations for task, total_time in time_logs.items()
    }
    error_rate = error_count / total_iterations

    # Display benchmark summary
    print("\nBenchmark Summary:")
    print(f"Error count: {error_count}")
    print(f"Error rate: {error_rate:.2%}")
    print("Average processing times (seconds):", avg_time_logs)

    # Analyze context lengths
    context_lengths = np.array(context_lengths)
    mean_length = context_lengths.mean()
    std_length = context_lengths.std()

    print("\nContext length statistics:")
    print(f"Mean: {mean_length:.2f} characters")
    print(f"Standard Deviation: {std_length:.2f} characters")

    # Save dataset items to JSON file
    with open("benchmark_dataset.json", "w") as file:
        json.dump(dataset_items, file)

    # Return summary of results
    return {
        "error_count": error_count,
        "error_rate": error_rate,
        "avg_time_per_task": avg_time_logs,
        "context_length_mean": mean_length,
        "context_length_std": std_length,
    }


# Run benchmark
benchmark_results = benchmark_challenger(
    n_iterations=20000,
    max_characters=10000,
    model_name="Condense-AI/Mistral-7B-Instruct-v0.2",
)
print("\nBenchmarking completed. Results:", benchmark_results)

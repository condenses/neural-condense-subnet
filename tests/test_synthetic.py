import time
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from neural_condense_core import validator_utils
import traceback
import asyncio


def benchmark_challenger(
    n_iterations: int = 10,
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
    if model_name == "universal":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    challenger = validator_utils.synthesizing.ChallengeGenerator(None)

    # Define task types and initialize logs
    tasks = [
        "question_answering",
        # "causal_conversation",
        # "reconstruct_conversation",
        # "trivial_qa_conversation",
    ]
    time_logs = {task: 0 for task in tasks}
    error_count = 0
    dataset_items = []
    context_lengths = []
    token_counts = []

    # Start progress bar for total iterations
    total_iterations = n_iterations * len(tasks)
    pbar = tqdm(total=total_iterations, desc="Benchmarking", unit="task")

    for i in range(n_iterations):
        for task in tasks:
            try:
                start_time = time.time()

                # Generate protocol using Challenger
                protocol = asyncio.run(
                    challenger.generate_challenge(
                        model_name=model_name,
                        task=task,
                        max_context_length_in_chars=max_characters,
                    )
                )

                # Record details of the generated sample
                item = {
                    "task": task,
                    "id": i,
                    "data": protocol.validator_payload,
                    "model_id": model_name,
                    "max_characters": max_characters,
                }

                # Track time taken for task
                time_logs[task] += time.time() - start_time

                # Store context length and token count for analysis
                context_lengths.append(
                    len(item["data"]["task_data"]["formatted_context"])
                )
                tokens = tokenizer.encode(
                    item["data"]["task_data"]["formatted_context"]
                )
                token_counts.append(len(tokens))

                # Add item to dataset items
                dataset_items.append(item)

            except Exception as e:
                traceback.print_exc()
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

    # Analyze context lengths and tokens
    context_lengths = np.array(context_lengths)
    token_counts = np.array(token_counts)
    mean_length = context_lengths.mean()
    std_length = context_lengths.std()
    mean_tokens = token_counts.mean()
    std_tokens = token_counts.std()

    print("\nContext length statistics:")
    print(f"Mean: {mean_length:.2f} characters")
    print(f"Standard Deviation: {std_length:.2f} characters")
    print("\nToken count statistics:")
    print(f"Mean: {mean_tokens:.2f} tokens")
    print(f"Standard Deviation: {std_tokens:.2f} tokens")

    # Save dataset items to JSON file
    with open("synthetic_samples.json", "w") as file:
        json.dump(dataset_items, file)

    # Return expanded summary of results
    return {
        "error_count": error_count,
        "error_rate": error_rate,
        "avg_time_per_task": avg_time_logs,
        "context_length_mean": mean_length,
        "context_length_std": std_length,
        "token_count_mean": mean_tokens,
        "token_count_std": std_tokens,
    }


# Run benchmark
benchmark_results = benchmark_challenger(
    n_iterations=100,
    max_characters=40000,
    model_name="universal",
)

print("\nBenchmarking completed. Results:", benchmark_results)

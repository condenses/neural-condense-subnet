import os
import json
from convo_generator import ConvoGenerator
from scheduler import Scheduler

def create_sample_data(num_qa_sets: int = 5, num_conversations: int = 3, output_path: str = "initial_data.json"):
    """
    Create a sample data file with QA sets and conversations.
    
    Args:
        num_qa_sets: Number of QA sets to generate
        num_conversations: Number of conversations to generate
        output_path: Where to save the JSON file
    """
    # Initialize the generator and scheduler
    generator = ConvoGenerator(api_key=os.environ["CORCEL_API_KEY"])
    scheduler = Scheduler(generator=generator)
    
    data = {
        "qa_sets": [],
        "conversations": []
    }
    
    # Generate QA sets
    print(f"Generating {num_qa_sets} QA sets...")
    for i in range(num_qa_sets):
        try:
            context_seed = f"Context {i}"
            questions, answers, total_chars = generator.generate_qa_pairs(context_seed, num_questions=3)
            qa_set = {
                "questions": questions,
                "answers": answers,
                "total_chars": total_chars,
                "context_seed": context_seed,
                "messages": []
            }
            data["qa_sets"].append(qa_set)
            print(f"Generated QA set {i + 1}/{num_qa_sets}")
        except Exception as e:
            print(f"Error generating QA set {i}: {e}")
    
    # Generate conversations
    print(f"\nGenerating {num_conversations} conversations...")
    for i in range(num_conversations):
        try:
            context_seed = f"Topic {i}"
            messages, total_chars = generator.generate_conversation(context_seed)
            conversation = {
                "messages": messages,
                "total_chars": total_chars,
                "context_seed": context_seed,
                "questions": [],
                "answers": []
            }
            data["conversations"].append(conversation)
            print(f"Generated conversation {i + 1}/{num_conversations}")
        except Exception as e:
            print(f"Error generating conversation {i}: {e}")
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved {len(data['qa_sets'])} QA sets and {len(data['conversations'])} conversations to {output_path}")
    return data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate initial data for the Scheduler')
    parser.add_argument('--qa-sets', type=int, default=5, help='Number of QA sets to generate')
    parser.add_argument('--conversations', type=int, default=3, help='Number of conversations to generate')
    parser.add_argument('--output', type=str, default='initial_data.json', help='Output file path')
    parser.add_argument('--push_to_hub', type=str, default="Condense-AI/subnet-synthetic-dataset-v0.2", help='Hub ID to push to')
    args = parser.parse_args()
    
    data = create_sample_data(
        num_qa_sets=args.qa_sets,
        num_conversations=args.conversations,
        output_path=args.output
    )

    if args.push_to_hub:
        from datasets import Dataset
        dataset = Dataset.from_dict(data)
        dataset.push_to_hub(args.push_to_hub)
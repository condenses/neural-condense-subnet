from datasets import Dataset
import pandas as pd
import json


data = json.load(open("benchmark_dataset.json"))

dataset = Dataset.from_pandas(pd.DataFrame(data))

dataset.push_to_hub("Condense-AI/benchmark-condense-v0.1")

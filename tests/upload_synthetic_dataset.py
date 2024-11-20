from datasets import Dataset
import pandas as pd
import json


data = json.load(open("synthetic_samples.json"))

dataset = Dataset.from_pandas(pd.DataFrame(data))

dataset.push_to_hub("Condense-AI/synthetic-samples-v0.1")

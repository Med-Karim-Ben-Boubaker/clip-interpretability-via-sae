from datasets import load_dataset
import dotenv
import os

dotenv.load_dotenv()

# Stream the dataset without downloading it entirely
dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="train",
    streaming=True,
    token=os.getenv("HF_TOKEN")
)

# Iterate over the dataset
for sample in dataset:
    print(sample)
    break  # Process one sample for demonstration


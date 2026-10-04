import huggingface_hub
print(f"huggingface_hub: {huggingface_hub.__version__}")

import datasets

print(f"datasets: {datasets.__version__}")

from datasets import load_dataset

print("¡Todo funciona correctamente!")

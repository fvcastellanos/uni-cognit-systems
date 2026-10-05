import json
import numpy as np
import pandas as pd
import torch
import datasets
datasets.config.TORCHVISION_AVAILABLE = False
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix

model_path = "/Users/fvcg/projects/uni-cognit-systems/train/banking77/distilroberta-banking77-saved"
info_path = "/Users/fvcg/.cache/huggingface/datasets/PolyAI___banking77/default/1.1.0/17ffc2ed47c2ed928bee64127ff1dbc97204cb974c2f980becae7c864007aed9/dataset_info.json"
names = json.load(open(info_path))["features"]["label"]["names"]

test_df = pd.read_csv("/Users/fvcg/projects/uni-cognit-systems/datasets/banking77/test.csv")
test_ds = Dataset.from_pandas(test_df[["text", "label"]])

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)


test_ds = test_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.rename_column("label", "labels")
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device, flush=True)
model.to(device)
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for i in range(0, len(test_ds), 32):
        batch = test_ds[i:i + 32]
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        all_preds.extend(out.logits.argmax(-1).cpu().numpy().tolist())
        all_labels.extend(batch["labels"].cpu().numpy().tolist())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
cm = confusion_matrix(all_labels, all_preds, labels=range(77))
cm_nod = cm.copy()
np.fill_diagonal(cm_nod, 0)

worst = [72, 74, 37, 62, 5, 22, 10]
print("=== Patrones de confusion por clase (fila=real, col=predicha) ===", flush=True)
for lbl in worst:
    print(f"\n--- {names[lbl]} (label {lbl}) ---", flush=True)
    fn = cm_nod[lbl]
    for t in np.argsort(-fn)[:6]:
        if fn[t] > 0:
            print(f"    FN -> {names[t]}: {fn[t]}", flush=True)
    fp = cm_nod[:, lbl]
    for t in np.argsort(-fp)[:6]:
        if fp[t] > 0:
            print(f"    FP <- {names[t]}: {fp[t]}", flush=True)

print("\n=== Top 20 pares mas confundidos (global) ===", flush=True)
pairs = []
for a in range(77):
    for b in range(77):
        if a != b and cm_nod[a, b] > 0:
            pairs.append((cm_nod[a, b], names[a], names[b]))
pairs.sort(reverse=True)
for n, a, b in pairs[:20]:
    print(f"{n:3d}  real={a:35s} pred={b}", flush=True)

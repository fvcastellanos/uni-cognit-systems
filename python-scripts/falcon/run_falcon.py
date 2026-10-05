import os
import re
import json
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "tiiuae/falcon-7b-instruct"
CACHE_DIR = "/Users/fvcg/projects/uni-cognit-systems/train/models/falcon-7b-instruct"
SAMPLES_CSV = "/Users/fvcg/projects/uni-cognit-systems/datasets/banking77/misclassified_samples.csv"
OUT_CSV = "/Users/fvcg/projects/uni-cognit-systems/datasets/banking77/falcon_justifications.csv"
CALIB_OUT = "/Users/fvcg/projects/uni-cognit-systems/datasets/banking77/falcon_calibration.json"


def buildExplanationPrompt(query, true_label, pred_label):
    return (
        'User: Eres un analista experto en clasificación de intenciones bancarias. '
        'Tu tarea es explicar, de forma breve y objetiva, por qué un clasificador '
        'automático se equivocó al etiquetar la siguiente consulta de un cliente.\n'
        '\n'
        'Datos del caso:\n'
        f'- Consulta del cliente: "{query}"\n'
        f'- Etiqueta correcta (real): {true_label}\n'
        f'- Etiqueta predicha por el modelo (incorrecta): {pred_label}\n'
        '\n'
        'Instrucciones:\n'
        f'1. Explica por qué el modelo pudo haber predicho "{pred_label}" en lugar de "{true_label}".\n'
        '2. Basa la explicación únicamente en la consulta (vocabulario compartido, '
        'ambigüedad, falta de contexto o matices sutiles).\n'
        '3. Responde en un MÁXIMO de dos oraciones, en prosa continua.\n'
        '4. No inventes información que no esté en la consulta ni menciones el nombre del modelo.\n'
        '\n'
        'Respuesta:\n'
        'Assistant:'
    )


def truncar_a_dos_oraciones(texto):
    texto = texto.strip()
    partes = re.split(r'(?<=[.!?])\s+', texto)
    res = " ".join(partes[:2]).strip()
    if res and res[-1] not in ".!?":
        res += "."
    return res


def log(msg):
    print(msg, flush=True)


log("Cargando tokenizer Falcon...")
falconTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

log("Cargando modelo Falcon (fp16)...")
falconModel = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    device_map="mps",  # forzar MPS: "auto" descarga capas a disco en Apple Silicon
    torch_dtype=torch.float16,
)
falconGenerator = pipeline(
    "text-generation",
    model=falconModel,
    tokenizer=falconTokenizer,
)


def generar(prompt, temperature=0.2, max_new_tokens=80):
    out = falconGenerator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=falconTokenizer.eos_token_id,
        return_full_text=False,
    )
    return truncar_a_dos_oraciones(out[0]["generated_text"])


samples = pd.read_csv(SAMPLES_CSV)
log(f"Cargadas {len(samples)} muestras mal clasificadas.")

# --- Calibración de temperatura sobre 2 muestras ---
calib = {}
temps = [0.0, 0.2, 0.7]
for idx in [0, 1]:
    row = samples.iloc[idx]
    p = buildExplanationPrompt(row["text"], row["true_label"], row["pred_label"])
    key = f"sample_{idx}"
    calib[key] = {
        "text": row["text"],
        "true_label": row["true_label"],
        "pred_label": row["pred_label"],
    }
    for t in temps:
        log(f"[calib] sample {idx} temp {t} ...")
        try:
            calib[key][f"temp_{t}"] = generar(p, temperature=t)
        except Exception as e:
            calib[key][f"temp_{t}"] = f"ERROR: {e}"
        log(f"[calib] sample {idx} temp {t} -> {calib[key][f'temp_{t}']}")

with open(CALIB_OUT, "w") as f:
    json.dump(calib, f, indent=2, ensure_ascii=False)
log(f"Calibración guardada en {CALIB_OUT}")

# --- Inferencia sobre las 20 muestras (temperatura fija 0.2) ---
results = []
for i, row in samples.iterrows():
    p = buildExplanationPrompt(row["text"], row["true_label"], row["pred_label"])
    log(f"[{i+1}/{len(samples)}] explicando... (true={row['true_label']} -> pred={row['pred_label']})")
    try:
        expl = generar(p, temperature=0.2)
    except Exception as e:
        expl = f"ERROR: {e}"
    results.append({
        "text": row["text"],
        "true_label": row["true_label"],
        "pred_label": row["pred_label"],
        "confidence": float(row["confidence"]),
        "explicacion": expl,
    })
    log(f"   -> {expl}")

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_CSV, index=False)
log(f"Justificaciones guardadas en {OUT_CSV}")
log("DONE")

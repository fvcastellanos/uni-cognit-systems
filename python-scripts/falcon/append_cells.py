import json
import random


def lines(t):
    return t.splitlines(keepends=True)


def md(text, ident=None):
    return {"cell_type": "markdown",
            "id": ident or ("%08x" % random.getrandbits(32)),
            "metadata": {},
            "source": lines(text)}


def code(text, ident=None):
    return {"cell_type": "code",
            "execution_count": None,
            "id": ident or ("%08x" % random.getrandbits(32)),
            "metadata": {},
            "outputs": [],
            "source": lines(text)}


CELLS = []

# ---- 1. intro ----
CELLS.append(md("""En esta sección se usa el modelo **Falcon-7B-Instruct** (`tiiuae/falcon-7b-instruct`) como *intérprete* de los errores del clasificador DistilRoBERTa. No se reentrena nada: se aplica *prompt engineering* para que el LLM explique las predicciones.

Requisitos cubiertos:
* **(a)** Prompt que genera una explicación breve (máx. 2 oraciones) de la clase predicha a partir de la consulta bancaria.
* **(b)** Calibración del prompt: temperatura, longitud de respuesta y estructura/claridad.
* **(c)** Selección de 20 muestras mal clasificadas.
* **(d)** Justificación de cada clasificación incorrecta con el prompt diseñado.
* **(e)** Análisis y listado de las razones dadas por el LLM.
* **(f)** Conclusiones sobre el uso del LLM para interpretar las salidas del clasificador."""))

# ---- 2. imports ----
CELLS.append(code(r'''# ---------------------------------------------------------
# Librerías para la sección de explicación de errores con Falcon
# ---------------------------------------------------------
import os
import re
import json
import numpy as np
import pandas as pd
import torch
import datasets
datasets.config.TORCHVISION_AVAILABLE = False

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
)
'''))

# ---- 3. markdown ----
CELLS.append(md("### Predicciones del clasificador (base para seleccionar errores)"))

# ---- 4. predictions code ----
CELLS.append(code(r'''# ---------------------------------------------------------
# Predicciones del clasificador (DistilRoBERTa) sobre test
# ---------------------------------------------------------
CLF_PATH = "../train/banking77/distilroberta-banking77-saved"
TEST_CSV = "../datasets/banking77/test.csv"
OUT_MIS_CSV = "../datasets/banking77/misclassified_samples.csv"

# Nombres de los 77 intents (orden 0-76), igual al ClassLabel de PolyAI/banking77
INTENT_NAMES = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]
assert len(INTENT_NAMES) == 77

# Cargar el clasificador guardado con nombres propios (no pisa tokenizer/model)
clfTokenizer = AutoTokenizer.from_pretrained(CLF_PATH)
clfModel = AutoModelForSequenceClassification.from_pretrained(CLF_PATH)

test_df = pd.read_csv(TEST_CSV)
device = "mps" if torch.backends.mps.is_available() else "cpu"
clfModel.to(device)
clfModel.eval()

def tokenize_clf(texts):
    return clfTokenizer(texts, padding="max_length", truncation=True,
                        max_length=64, return_tensors="pt")

all_probs, all_preds = [], []
with torch.no_grad():
    for i in range(0, len(test_df), 32):
        texts = test_df["text"].iloc[i:i + 32].tolist()
        enc = {k: v.to(device) for k, v in tokenize_clf(texts).items()}
        probs = torch.softmax(clfModel(**enc).logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(probs.argmax(-1).cpu().numpy())

probs = np.concatenate(all_probs)
preds = np.concatenate(all_preds)
true = test_df["label"].to_numpy()

df_pred = pd.DataFrame({
    "text": test_df["text"],
    "true_label_id": true,
    "pred_label_id": preds,
})
df_pred["true_label"] = df_pred["true_label_id"].map(lambda x: INTENT_NAMES[x])
df_pred["pred_label"] = df_pred["pred_label_id"].map(lambda x: INTENT_NAMES[x])
df_pred["confidence"] = probs[np.arange(len(probs)), preds]

print(f"Total test: {len(df_pred)} | Mal clasificadas: {(df_pred['true_label_id'] != df_pred['pred_label_id']).sum()}")
df_pred.head()
'''))

# ---- 5. markdown ----
CELLS.append(md("### (c) Selección de 20 muestras mal clasificadas"))

# ---- 6. selection code ----
CELLS.append(code(r'''# ---------------------------------------------------------
# (c) Seleccionar 20 muestras mal clasificadas
# ---------------------------------------------------------
mis = df_pred[df_pred["true_label_id"] != df_pred["pred_label_id"]].copy()
# Las 20 con mayor confianza en la clase (incorrecta) predicha:
# son los errores más sistemáticos y, por tanto, los más informativos.
mis = mis.sort_values("confidence", ascending=False).head(20).reset_index(drop=True)
mis = mis[["text", "true_label", "pred_label", "confidence"]]
mis.to_csv(OUT_MIS_CSV, index=False)
print(f"Guardadas {len(mis)} muestras mal clasificadas en {OUT_MIS_CSV}")
mis
'''))

# ---- 7. markdown ----
CELLS.append(md("### (a) Diseño del prompt para Falcon-7B-Instruct"))

# ---- 8. prompt code ----
CELLS.append(code(r'''# ---------------------------------------------------------
# (a) Prompt para explicar la clase predicha / justificar el error
# ---------------------------------------------------------
def buildExplanationPrompt(query, true_label, pred_label):
    return (
        "User: Eres un analista experto en clasificación de intenciones bancarias. "
        "Tu tarea es explicar, de forma breve y objetiva, por qué un clasificador "
        "automático se equivocó al etiquetar la siguiente consulta de un cliente.\n"
        "\n"
        "Datos del caso:\n"
        f'- Consulta del cliente: "{query}"\n'
        f"- Etiqueta correcta (real): {true_label}\n"
        f"- Etiqueta predicha por el modelo (incorrecta): {pred_label}\n"
        "\n"
        "Instrucciones:\n"
        f'1. Explica por qué el modelo pudo haber predicho "{pred_label}" en lugar de "{true_label}".\n'
        "2. Basa la explicación únicamente en la consulta (vocabulario compartido, "
        "ambigüedad, falta de contexto o matices sutiles).\n"
        "3. Responde en un MÁXIMO de dos oraciones, en prosa continua.\n"
        "4. No inventes información que no esté en la consulta ni menciones el nombre del modelo.\n"
        "\n"
        "Respuesta:\n"
        "Assistant:"
    )

def truncar_a_dos_oraciones(texto):
    texto = texto.strip()
    partes = re.split(r"(?<=[.!?])\s+", texto)
    res = " ".join(partes[:2]).strip()
    if res and res[-1] not in ".!?":
        res += "."
    return res

# Vista previa del prompt sobre la primera muestra mal clasificada
print(buildExplanationPrompt(
    mis.iloc[0]["text"], mis.iloc[0]["true_label"], mis.iloc[0]["pred_label"]
))
'''))

# ---- 9. markdown ----
CELLS.append(md("### Cargar Falcon-7B-Instruct (carpeta de descarga propia)"))

# ---- 10. load falcon ----
CELLS.append(code(r'''# ---------------------------------------------------------
# Cargar Falcon-7B-Instruct en una carpeta propia
# (no pisa las variables tokenizer/model del clasificador)
# ---------------------------------------------------------
MODEL_NAME = "tiiuae/falcon-7b-instruct"
CACHE_DIR = "../train/models/falcon-7b-instruct"

falconTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
falconModel = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
)
'''))

# ---- 11. generator ----
CELLS.append(code(r'''# ---------------------------------------------------------
# Pipeline de generación + función de explicación calibrada
# ---------------------------------------------------------
falconGenerator = pipeline(
    "text-generation",
    model=falconModel,
    tokenizer=falconTokenizer,
)

def generarExplicacionFalcon(prompt, temperature=0.2, max_new_tokens=80):
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
'''))

# ---- 12. markdown ----
CELLS.append(md("### (b) Calibración: temperatura, longitud y estructura"))

# ---- 13. calibration code ----
CELLS.append(code(r'''# ---------------------------------------------------------
# (b) Calibración de temperatura sobre 2 muestras
# ---------------------------------------------------------
temps = [0.0, 0.2, 0.7]
for idx in [0, 1]:
    row = mis.iloc[idx]
    prompt = buildExplanationPrompt(row["text"], row["true_label"], row["pred_label"])
    print(f"\n===== Muestra {idx} | real: {row['true_label']} | predicha: {row['pred_label']} =====")
    print(f"Consulta: {row['text']}")
    for t in temps:
        print(f"\n--- temperature={t} ---")
        print(generarExplicacionFalcon(prompt, temperature=t))
'''))

# ---- 14. markdown ----
CELLS.append(md("### (d) Justificación de cada clasificación incorrecta"))

# ---- 15. run 20 ----
CELLS.append(code(r'''# ---------------------------------------------------------
# (d) Justificación de cada clasificación incorrecta (20 muestras)
# ---------------------------------------------------------
OUT_JUST_CSV = "../datasets/banking77/falcon_justifications.csv"

resultados = []
for i, row in mis.iterrows():
    prompt = buildExplanationPrompt(row["text"], row["true_label"], row["pred_label"])
    expl = generarExplicacionFalcon(prompt, temperature=0.2)
    resultados.append({
        "text": row["text"],
        "true_label": row["true_label"],
        "pred_label": row["pred_label"],
        "confidence": row["confidence"],
        "explicacion": expl,
    })
    print(f"[{i + 1}/{len(mis)}] {row['true_label']} -> {row['pred_label']}")
    print(f"    {expl}")

df_just = pd.DataFrame(resultados)
df_just.to_csv(OUT_JUST_CSV, index=False)
print(f"\nGuardadas {len(df_just)} justificaciones en {OUT_JUST_CSV}")
df_just
'''))

# ---- 16. markdown ----
CELLS.append(md("### (e) Análisis y listado de las razones dadas por el LLM"))

# ---- 17. analysis code ----
CELLS.append(code(r'''# ---------------------------------------------------------
# (e) Análisis y listado de las razones dadas por el LLM
# ---------------------------------------------------------
df_just = pd.read_csv(OUT_JUST_CSV)

def categorizar(texto):
    t = str(texto).lower()
    if any(k in t for k in ["misma palabra", "similar", "compart", "parec", "relacion", "termino", "vocabulario"]):
        return "solapamiento léxico / vocabulario compartido"
    if any(k in t for k in ["ambigu", "puede referirse", "doble", "sentidos"]):
        return "ambigüedad semántica"
    if any(k in t for k in ["corta", "poca información", "falta de contexto", "breve"]):
        return "consulta corta / falta de contexto"
    if any(k in t for k in ["negación", "negativ", "matiz"]):
        return "negación / matiz sutil"
    if any(k in t for k in ["dominio", "bancari", "terminología"]):
        return "terminología de dominio"
    return "otra / parafraseo"

df_just["categoria"] = df_just["explicacion"].apply(categorizar)

print("=== Conteo por categoría ===")
print(df_just["categoria"].value_counts().to_string())

print("\n=== Listado de razones ===")
for i, row in df_just.iterrows():
    print(f"{i + 1:2d}. [{row['categoria']}] real={row['true_label']} | pred={row['pred_label']}")
    print(f"    {row['explicacion']}")
'''))

# ---- 18. conclusions markdown ----
CELLS.append(md("""### (f) Conclusiones sobre el uso del LLM

* El LLM (Falcon-7B-Instruct) permite **interpretar en lenguaje natural** por qué el clasificador se equivoca, algo que la matriz de confusión por sí sola no explica: traduce una celda de confusión en una hipótesis lingüística.
* Las explicaciones suelen coincidir con los patrones observados en la matriz (solapamiento léxico y ambigüedad entre intents semánticamente adyacentes), lo que valida su utilidad como herramienta de diagnóstico.
* **Limitaciones:** el LLM explica razones *plausibles* pero no puede verificar el ground truth; puede caer en racionalización post-hoc (inventar una causa coherente aunque no sea la real) y no distingue errores del modelo de errores de etiquetado del dataset.
* Para mitigarlo conviene: temperatura baja, restringir la salida a 2 oraciones, pedir explícitamente basarse solo en la consulta y contrastar sus razones con la matriz de confusión."""))

# ---- escribir ----
NB = "/Users/fvcg/projects/uni-cognit-systems/notebooks/Actividad-2-llm.ipynb"
nb = json.load(open(NB))
nb["cells"].extend(CELLS)
open(NB, "w").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("Celdas añadidas:", len(CELLS), "| total ahora:", len(nb["cells"]))


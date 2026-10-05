import json

NB = "/Users/fvcg/projects/uni-cognit-systems/notebooks/Actividad-2-llm.ipynb"


def lines(t):
    return t.splitlines(keepends=True)


analysis_code = r'''# ---------------------------------------------------------
# (e) Análisis y listado de las razones dadas por el LLM
# ---------------------------------------------------------
df_just = pd.read_csv(OUT_JUST_CSV)

def categorizar(texto):
    t = str(texto).lower()
    if any(k in t for k in ["interchangeably", "similar", "common term", "sound", "same word", "misma palabra", "compart"]):
        return "solapamiento léxico / términos similares"
    if any(k in t for k in ["lack of context", "ambigu", "unclear", "more detail", "poca información", "falta de contexto"]):
        return "ambigüedad / falta de contexto"
    if any(k in t for k in ["apolog", "inconvenience", "understand", "concerned"]):
        return "respuesta empática (no explica el error)"
    if any(k in t for k in ["not properly trained", "technical", "programming", "algorithm", "designed to detect", "not perfect"]):
        return "atribución a error técnico / del modelo"
    if any(k in t for k in ["account balance", "credit or debit", "fee for the deposit"]):
        return "alucinación (inventa datos)"
    return "otra / especulativa"

df_just["categoria"] = df_just["explicacion"].apply(categorizar)

print("=== Conteo por categoría ===")
print(df_just["categoria"].value_counts().to_string())

print("\n=== Listado de razones ===")
for i, row in df_just.iterrows():
    print(f"{i + 1:2d}. [{row['categoria']}] real={row['true_label']} | pred={row['pred_label']}")
    print(f"    {row['explicacion']}")
'''

conclusions = """### (f) Conclusiones sobre el uso del LLM

**Calibración (temperatura):**
* `temperature=0.0` (greedy) a veces no sigue la instrucción y responde solo con empatía ("I apologize...") sin explicar el error.
* `temperature=0.2` ofrece el mejor equilibrio: explica el error de forma coherente y basada en la consulta.
* `temperature=0.7` añade especulación técnica no fundamentada ("programming error", "algorithm"), lo que aumenta el riesgo de alucinación.
* Por eso se fijó **temperature=0.2** con `top_p=0.9`, `max_new_tokens=80` y recorte a 2 oraciones.

**Valor del LLM como intérprete del clasificador:**
* Traduce cada error en una hipótesis en lenguaje natural. En varios casos acierta el mecanismo real, p. ej. `exchange_rate` vs `exchange_charge` ("términos usados indistintamente") o `card_not_working` vs `virtual_card_not_working` ("términos que suenan parecidos"), lo cual coincide con la matriz de confusión.

**Distribución de las razones (20 muestras):**
* ~9/20 → "ambigüedad / falta de contexto" (razón genérica pero plausible).
* 3/20 → "solapamiento léxico / términos similares".
* 3/20 → "alucinación" (inventa hechos no presentes en la consulta, p. ej. "saldo insuficiente", "tarjeta de crédito").
* 2/20 → "respuesta empática sin explicar el error".
* 2/20 → "atribución a error técnico / del modelo".

**Limitaciones observadas:**
* El LLM tiende a la **racionalización post-hoc**: ante la falta de información real, inventa causas coherentes (saldo insuficiente, error de programación) aunque no estén en la consulta.
* La respuesta "falta de contexto" es un **comodín** que no siempre identifica la causa verdadera (a menudo es solapamiento léxico, no falta de contexto).
* El LLM **no puede verificar el ground truth**: ofrece razones plausibles, no la causa real del error del clasificador.

**Recomendaciones de uso:**
* Mantener temperatura baja (0.2) y pedir explícitamente basarse solo en la consulta.
* Contrastar las explicaciones con la matriz de confusión para descartar alucinaciones.
* Usar el LLM como complemento del análisis cuantitativo (F1 por clase, pares de confusión), no como única fuente de diagnóstico."""

nb = json.load(open(NB))
nb["cells"][56]["source"] = lines(analysis_code)
nb["cells"][57]["source"] = lines(conclusions)
open(NB, "w").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("Celdas 56 y 57 actualizadas.")

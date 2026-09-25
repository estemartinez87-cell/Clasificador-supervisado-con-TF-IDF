# -*- coding: utf-8 -*-
"""
Checkpoint Módulo 3: Clasificador Supervisado con TF-IDF
Dataset: AG News (ag_news_train.csv / ag_news_test.csv — mismo corpus del Módulo 2).

Estructura (mapeada a la consigna):
  1. Carga de datos
  2. Reutilización del preprocesamiento del Módulo 2 (Regex + lematización SpaCy)
  3. Vectorización TF-IDF (experimentando max_features y ngram_range)
  4. Modelado (comparación de clasificadores, se elige uno como baseline)
  5. Evaluación (classification_report + matriz de confusión)

Sin Data Leakage: fit_transform del TfidfVectorizer SOLO sobre train,
transform sobre test.
"""

import re
import html
import json
import time
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Carga de datos (mismo corpus del Módulo 2 — ag_news_train.csv / ag_news_test.csv)
# ---------------------------------------------------------------------------
train_df = pd.read_csv("ag_news_train.csv")
test_df = pd.read_csv("ag_news_test.csv")
print(f"Train: {train_df.shape} | Test: {test_df.shape}")
print("Clases:", sorted(train_df["label"].unique()))

# ---------------------------------------------------------------------------
# 2. Reutilización del preprocesamiento del Módulo 2 (Regex + lematización SpaCy)
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOPWORDS = nlp.Defaults.stop_words


def clean_text(text: str) -> str:
    """Limpieza Regex: decodifica HTML, quita tags/URLs y caracteres no
    alfanuméricos. Incluye el fix para la entidad 'quot' mal codificada
    que detectamos en el EDA del Módulo 2."""
    text = html.unescape(text)
    text = re.sub(r"#39;", "'", text)
    text = re.sub(r"\bquot\b", " ", text)  # fix: entidad &quot; mal codificada (ver EDA M2)
    text = re.sub(r"&lt;.*?&gt;", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("\\", "")
    text = re.sub(r"[^A-Za-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """Función pedida por la consigna del Módulo 2: limpieza + normalización +
    lematización, integradas en un solo paso."""
    cleaned = clean_text(text).lower()
    doc = nlp(cleaned)
    tokens = [
        tok.lemma_
        for tok in doc
        if not tok.is_punct and not tok.is_space and tok.lemma_.strip() != ""
    ]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def batch_preprocess(texts, remove_stopwords=True):
    """Versión batch (vía nlp.pipe) para procesar todo el corpus rápido."""
    cleaned = [clean_text(t).lower() for t in texts]
    out = []
    for doc in nlp.pipe(cleaned, batch_size=200):
        tokens = [
            tok.lemma_
            for tok in doc
            if not tok.is_punct and not tok.is_space and tok.lemma_.strip() != ""
        ]
        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        out.append(" ".join(tokens))
    return out


print("\nAplicando preprocess_text sobre train...")
t0 = time.time()
train_df["clean_text"] = batch_preprocess(train_df["text"].tolist())
print(f"  {time.time() - t0:.1f}s")

print("Aplicando preprocess_text sobre test...")
t0 = time.time()
test_df["clean_text"] = batch_preprocess(test_df["text"].tolist())
print(f"  {time.time() - t0:.1f}s")

X_train_text, y_train = train_df["clean_text"], train_df["label"]
X_test_text, y_test = test_df["clean_text"], test_df["label"]

# ---------------------------------------------------------------------------
# 3. Vectorización TF-IDF — SIN Data Leakage:
#    fit_transform SOLO en train, transform en test.
#    Experimentación pedida por la consigna: max_features y ngram_range.
# ---------------------------------------------------------------------------
configs = [
    {"name": "unigrama, sin límite de vocab", "max_features": None, "ngram_range": (1, 1)},
    {"name": "unigrama, max_features=5000", "max_features": 5000, "ngram_range": (1, 1)},
    {"name": "uni+bigrama, sin límite de vocab", "max_features": None, "ngram_range": (1, 2)},
    {"name": "uni+bigrama, max_features=5000", "max_features": 5000, "ngram_range": (1, 2)},
]

print("\n=== Experimentación de vectorización (con LogisticRegression fijo) ===")
vec_results = []
for cfg in configs:
    vectorizer = TfidfVectorizer(max_features=cfg["max_features"], ngram_range=cfg["ngram_range"])
    Xtr = vectorizer.fit_transform(X_train_text)  # fit SOLO en train
    Xte = vectorizer.transform(X_test_text)       # transform en test (sin fit)

    clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)

    f1_macro = f1_score(y_test, preds, average="macro")
    acc = accuracy_score(y_test, preds)
    print(f"  {cfg['name']:<35} | vocab={Xtr.shape[1]:>6} | acc={acc:.4f} | F1-macro={f1_macro:.4f}")
    vec_results.append({**cfg, "vocab_size": Xtr.shape[1], "accuracy": acc, "f1_macro": f1_macro})

best_cfg = max(vec_results, key=lambda r: r["f1_macro"])
print(f"\nMejor configuración de vectorización: {best_cfg['name']} (F1-macro={best_cfg['f1_macro']:.4f})")

# ---------------------------------------------------------------------------
# 4. Modelado — comparación de clasificadores con la MEJOR configuración de TF-IDF
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=best_cfg["max_features"], ngram_range=best_cfg["ngram_range"])
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE),
    "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
}

print("\n=== Comparación de modelos ===")
model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    model_results[name] = {"model": model, "preds": preds, "accuracy": acc, "f1_macro": f1_macro}
    print(f"  {name:<22} | acc={acc:.4f} | F1-macro={f1_macro:.4f}")

best_model_name = max(model_results, key=lambda k: model_results[k]["f1_macro"])
best_model = model_results[best_model_name]["model"]
best_preds = model_results[best_model_name]["preds"]
print(f"\nModelo elegido como baseline final: {best_model_name}")

# ---------------------------------------------------------------------------
# 5. Evaluación: classification_report + matriz de confusión
# ---------------------------------------------------------------------------
report = classification_report(y_test, best_preds, digits=4)
print(f"\n=== Classification Report ({best_model_name}) ===")
print(report)

with open("reports/classification_report.txt", "w") as f:
    f.write(f"Modelo: {best_model_name}\n")
    f.write(f"Vectorizador: TF-IDF ({best_cfg['name']}, max_features={best_cfg['max_features']}, ngram_range={best_cfg['ngram_range']})\n\n")
    f.write(report)

labels = sorted(y_train.unique())
cm = confusion_matrix(y_test, best_preds, labels=labels)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=30)
plt.title(f"Matriz de confusión — {best_model_name}\n(TF-IDF: {best_cfg['name']})")
plt.tight_layout()
plt.savefig("reports/figures/confusion_matrix.png", dpi=150)
plt.close()

# Análisis de errores más frecuentes (para el README)
off_diag_errors = []
for i, true_label in enumerate(labels):
    for j, pred_label in enumerate(labels):
        if i != j and cm[i, j] > 0:
            off_diag_errors.append((true_label, pred_label, int(cm[i, j])))
off_diag_errors.sort(key=lambda x: -x[2])

print("\nTop confusiones (real -> predicho : cantidad):")
for true_l, pred_l, count in off_diag_errors[:5]:
    print(f"  {true_l} -> {pred_l} : {count}")

# Resumen estructurado
summary = {
    "best_vec_config": {k: v for k, v in best_cfg.items()},
    "vec_experiments": vec_results,
    "model_comparison": {
        k: {"accuracy": v["accuracy"], "f1_macro": v["f1_macro"]} for k, v in model_results.items()
    },
    "best_model": best_model_name,
    "top_confusions": off_diag_errors[:8],
    "n_train": len(train_df),
    "n_test": len(test_df),
}
with open("reports/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nListo. Reporte, matriz de confusión y resumen guardados en reports/.")

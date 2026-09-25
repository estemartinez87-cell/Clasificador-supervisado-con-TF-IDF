# Checkpoint Módulo 3 — Clasificador Supervisado con TF-IDF

Pipeline de clasificación de texto sobre **AG News** (4 clases: Business, Sci_Tech, Sports, World), usando vectorización TF-IDF y clasificadores clásicos de Scikit-Learn. Reutiliza el preprocesamiento (limpieza Regex + lematización SpaCy) desarrollado en el Módulo 2.

## Dataset

Mismo corpus del Módulo 2: `ag_news_train.csv` (8,000 ejemplos) y `ag_news_test.csv` (2,000 ejemplos), con columnas `text` y `label`. El test está perfectamente balanceado (500 ejemplos por clase).

## Pipeline

1. **Carga de datos** desde los CSV provistos.
2. **Preprocesamiento** (reutilizado del Módulo 2): decodificación HTML, limpieza regex, normalización a minúsculas, lematización con spaCy y remoción de stopwords. Se sumó un fix puntual detectado en el EDA del Módulo 2: la entidad `&quot;` aparecía mal codificada como la palabra suelta `"quot"` en el corpus; se agregó una regla para eliminarla antes de vectorizar.
3. **Vectorización TF-IDF**, evitando *data leakage*: `fit_transform` solo sobre train, `transform` sobre test. Se experimentó con 4 configuraciones (`max_features` y `ngram_range`):

   | Configuración | Vocabulario | Accuracy | F1-macro |
   |---|---|---|---|
   | **Unigrama, sin límite de vocab** | **17,510** | **0.8985** | **0.8984** |
   | Unigrama, max_features=5000 | 5,000 | 0.8945 | 0.8945 |
   | Uni+bigrama, sin límite de vocab | 154,539 | 0.8965 | 0.8963 |
   | Uni+bigrama, max_features=5000 | 5,000 | 0.8945 | 0.8945 |

   **Mejor configuración**: unigrama sin límite de vocabulario. Agregar bigramas no mejora el resultado con este corpus — el poder discriminativo ya está en las palabras individuales.

4. **Modelado**: comparación de 3 clasificadores sobre la mejor configuración de TF-IDF.

   | Modelo | Accuracy | F1-macro |
   |---|---|---|
   | **Logistic Regression** | **0.8985** | **0.8984** |
   | Naive Bayes | 0.8970 | 0.8973 |
   | Linear SVM | 0.8930 | 0.8929 |

## Justificación del modelo elegido

Se eligió **Logistic Regression** como baseline final. Los tres modelos quedaron muy cerca (diferencia de 0.5 a 1.1 puntos de F1-macro entre el mejor y el peor), lo cual ya es en sí mismo un dato relevante: con esta representación (TF-IDF + unigramas), el techo de rendimiento es similar para los tres algoritmos, y la elección del modelo aporta menos que la elección del vectorizador. Dicho esto, Logistic Regression tiene ventajas concretas frente a los otros dos en este caso:

- Superó a Naive Bayes en F1-macro (0.8984 vs. 0.8973), sin el supuesto de independencia condicional entre features que Naive Bayes necesita y que rara vez se cumple del todo en texto real.
- A diferencia de Linear SVM, entrega probabilidades calibrables (`predict_proba`), útiles si más adelante se necesita un umbral de confianza o se integra en un sistema que pondere la certeza de la predicción.
- Es más rápido de entrenar e interpretar que SVM sobre una matriz de ~17,500 features, y sus coeficientes por clase son directamente interpretables (qué palabras empujan hacia cada categoría), lo cual es valioso para explicar el modelo si hace falta.

## Vectorizador — parámetros finales

```python
TfidfVectorizer(max_features=None, ngram_range=(1, 1))
```

Sin límite de vocabulario y solo unigramas: la experimentación mostró que ni limitar `max_features` ni agregar bigramas mejora el F1-macro en este corpus, así que se optó por la configuración más simple entre las que dieron mejor resultado.

## Reporte de clasificación (Logistic Regression)

| Clase | Precision | Recall | F1-score |
|---|---|---|---|
| Business | 0.8484 | 0.8620 | 0.8552 |
| Sci_Tech | 0.8763 | 0.8640 | 0.8701 |
| Sports | 0.9548 | 0.9720 | 0.9633 |
| World | 0.9143 | 0.8960 | 0.9051 |
| **Accuracy** | | | **0.8985** |

## Análisis preliminar — ¿qué categorías son más difíciles?

Según la matriz de confusión (`reports/figures/confusion_matrix.png`), las confusiones más frecuentes son:

| Real | Predicho | Cantidad |
|---|---|---|
| Sci_Tech | Business | 47 |
| Business | Sci_Tech | 40 |
| Business | World | 24 |
| World | Business | 24 |
| World | Sci_Tech | 17 |

**Business es la clase más difícil** de predecir (menor F1-score, 0.8552), y se confunde en ambas direcciones con Sci_Tech y con World. Esto tiene sentido de dominio: muchas noticias de negocios mencionan empresas tecnológicas (confusión con Sci_Tech) o tienen componente geopolítico — acuerdos comerciales, sanciones, negociaciones internacionales (confusión con World). **Sports es la clase mejor resuelta** (F1=0.9633), porque su vocabulario es el más distintivo y autocontenido del dataset (nombres de equipos, resultados, terminología deportiva específica que no aparece en las otras categorías).

## Estructura del repositorio

```
├── Checkpoint_M3_TFIDF.ipynb   # notebook ejecutado de punta a punta, sin errores
├── baseline_tfidf.py           # mismo pipeline como script
├── README.md
├── requirements.txt
├── ag_news_train.csv / ag_news_test.csv
└── reports/
    ├── classification_report.txt
    ├── summary.json
    └── figures/
        └── confusion_matrix.png
```

## Cómo correrlo

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python baseline_tfidf.py
```

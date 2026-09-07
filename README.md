# Procesamiento del Lenguaje Natural

**Universidad Nacional de Colombia — Sede Manizales** · Facultad de Ingeniería y Arquitectura

> **Página oficial del curso:**
> ## [UN-GCPDS.github.io/ProcesamientoLenguajeNatural](https://UN-GCPDS.github.io/ProcesamientoLenguajeNatural/)
>
> El repositorio contiene el material del curso organizado por clases: notebooks, presentaciones y recursos de apoyo.

## Descripción del curso

La asignatura abarca desde los fundamentos del procesamiento de texto con Python, expresiones regulares, Pandas y SpaCy, pasando por la representación vectorial (BoW, TF-IDF, GloVe y BERT) y los modelos secuenciales (RNN, LSTM y GRU), hasta Transformers, generación con GPT-2, RAG con FAISS y Gemini, y el despliegue de agentes con n8n y Streamlit.

Se requiere manejo básico de Python. No se requiere experiencia previa en lingüística computacional.

## Ruta de aprendizaje

El contenido se encuentra publicado en la [página del curso](https://UN-GCPDS.github.io/ProcesamientoLenguajeNatural/) y se organiza en tres unidades, las cuales se desarrollan en orden secuencial:

| Unidad | Contenidos | Clases |
|---|---|---|
| **1 · Fundamentos del PLN** | Strings y slicing en Python · Regex avanzado, Pandas y normalización Unicode · Pipeline NLP con SMS Spam, SpaCy y POS tagging · De palabras a vectores: One-Hot, BoW, TF-IDF, GloVe y BERT | 1–4 |
| **2 · Deep Learning y LLMs** | Modelos secuenciales RNN, LSTM y GRU en PyTorch · Transformers: atención y positional encoding · Sabores Encoder, Decoder y Encoder-Decoder · Decoding autoregresivo con GPT-2 · RAG con embeddings, FAISS y Gemini | 5–9 |
| **3 · Agentes y despliegue MLOps** | Agente de logística con n8n e inventario UNAL · Frontend Streamlit con túnel Ngrok y reportes en PDF | 10–11 |

Cada página de clase incluye la fundamentación teórica, el notebook correspondiente y los enlaces al código fuente.

## Recursos por unidad

Los enlaces y recursos se encuentran disponibles en la barra lateral de la página de cada unidad:

- **Unidad 1:** pipeline SMS Spam (`Insumos/SMSSpamCollection.txt`), complementario de expresiones regulares y quizzes de las Clases 3 y 7. La unidad se evalúa con el **Parcial 1**.
- **Unidad 2:** notebooks de PyTorch/Transformers/RAG con GPU T4 en Colab, API de Gemini y FAISS. La unidad se evalúa con el **Taller 1** (chatbot Car-ing is sharing).
- **Unidad 3:** presentaciones en PDF (`Clase 10` y `Clase 11`), `Peticion.py` (webhook n8n), inventario de laboratorio y artefacto Streamlit. La unidad es de carácter práctico y se aprueba con el despliegue end-to-end.

## Evaluaciones

- **Parcial 1** (Unidad 1): regex con grupos nombrados, limpieza con SpaCy, features manuales, BoW/TF-IDF y GloVe vs BERT.
- **Taller 1** (Unidad 2): prototipado de chatbot Car-ing is sharing — clasificación de sentimiento (accuracy/F1), traducción (BLEU), QA extractivo y resumen abstractivo.
- **Quizzes** (Clases 3, 7 y 9): formularios de Google de 14 preguntas enlazados desde cada clase.
- **Despliegue MLOps** (Unidad 3): agente n8n sobre inventario y frontend Streamlit con Ngrok.

Las fechas y los criterios de entrega son anunciados por el docente en clase.

## Material y notebooks

Cada carpeta `Clase N/` contiene el notebook (`.ipynb`) de la clase y, según corresponda, el quiz (`Quizz.md`), los insumos (`.xlsx`, `.pdf`) y la presentación (`.pdf`).

El acceso al material se realiza desde la página de cada clase mediante el botón **Explorar Clase**. Los archivos `.ipynb` pueden ejecutarse en Google Colab o JupyterLab. El código fuente se encuentra disponible en este repositorio.

## Preguntas frecuentes

- **Acceso al contenido:** la totalidad del contenido se encuentra en la [página del curso](https://UN-GCPDS.github.io/ProcesamientoLenguajeNatural/). Se recomienda iniciar por la Unidad 1.
- **Organización del repositorio:** el material se encuentra distribuido en carpetas `Clase N/`, además de `Taller 1/`, `Parcial 1/` e `Insumos/`, cada una con el contenido correspondiente a la sesión.
- **Requerimientos de hardware:** la Unidad 1 no requiere GPU. Para las Unidades 2 y 3 se recomienda Google Colab con GPU T4, clave de Gemini AI Studio y cuenta gratuita de Ngrok.
- **Quizzes:** son formularios de Google enlazados en las Clases 3, 7 y 9 para autoevaluación.

---

<details>
<summary><b>Información para colaboradores (despliegue)</b></summary>

El sitio web se encuentra en `Pagina/` (HTML estático con Tailwind mediante CDN) y se publica en GitHub Pages a través de [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) en cada push a `main`.

- Estructura: `index.html`, además de `unidad-1/`, `unidad-2/` y `unidad-3/` (cada una con `index.html` y sus archivos `clase-N.html`).
- Los archivos PDF de las presentaciones se copian durante la integración continua a `Pagina/assets/presentaciones/` (ver el workflow); por este motivo no se encuentran disponibles en el entorno local.
- El directorio `Pagina/mockup/` se excluye del artefacto publicado.

</details>

## Licencia

[MIT License](LICENSE) — © 2026 UNAL-Manizales / GCPDS.

# Taller 1 — Rúbrica de Evaluación para IA Evaluadora (local, NO subir a Classroom)
### Procesamiento del Lenguaje Natural — UNAL Sede Manizales
**Temario:** Unidad 2, Clases 5–9 (RNN/LSTM/GRU, Transformers encoder/decoder, robustez, XAI con atención) · **Archivo estudiante:** `Taller 1/notebook.ipynb`
**En Classroom va SOLO la instrucción** (este mismo notebook). La calificación vive aquí, en este `.md`, y la ejecuta una IA con el MCP `google-classroom`.

---

## 1. Reglas generales UNAL (obligatorias)

- Escala final **0.0 a 5.0**, con **un decimal** (ej: 4.3, 2.8). No usar letras.
- **Aprueba con >= 3.0**.
- Son **5 puntos, cada uno vale 1.0**. La nota final es **la suma directa**:
  ```
  final = round(E1 + E2 + E3 + E4 + E5, 1)
  donde cada Ei va de 0.0 a 1.0
  ```
- Si el estudiante no entrega un ejercicio: **0.0** en ese ejercicio, con `no_entregado: true`.
- Si el código **no corre** (SyntaxError, variable indefinida, import faltante) pero la lógica es correcta: máximo **0.5** en ese ejercicio. Indica el error en `justificacion`.
- Si hay **copia literal** de otro estudiante / de la solución de clase sin aporte propio: penaliza **0.2** en cada ejercicio copiado y márcalo en `observaciones`.
- Evalúa **lo que el estudiante entregó**, no lo que "quiso decir". No inventes código ni variables que no están. Si una variable obligatoria no existe, el chequeo correspondiente vale 0.
- Acepta español o inglés. No penalices ortografía ni estilo (nombres, `.copy()`, `apply` vs `.str`) cuando el output es correcto. Penaliza solo ineficiencias reales (bucles `for i in range(len(df))` con `iloc`, fuga train/test).
- **Fuga train/test = fallo grave:** si el vocabulario, el `pad_len` o cualquier estadística se calcula sobre todo el DF (incluido test), máximo **0.6** en E1 y arrastra la comparación (máximo 0.7 en E2/E3 con nota en observaciones).
- Los umbrales numéricos de este documento son de **plausibilidad** (formato/rango/tendencia), no de ranking entre estudiantes. No ordenes por accuracy.

### Niveles genéricos (cada Ei de 0.0 a 1.0)

| Rango Ei | Etiqueta | Descriptor |
|---|---|---|
| 1.0 | Excelente | Funciona, protocolo respetado, explica el porqué causal, sin errores. |
| 0.8 – 0.9 | Sobresaliente | Funciona con detalles menores (un borde, explicación incompleta). |
| 0.6 – 0.7 | Aprobatorio parcial | Idea correcta con errores parciales o explicación superficial. |
| 0.2 – 0.5 | Insuficiente | Intento serio con fallos graves: no corre, máscara mal aplicada, confunde conceptos. |
| 0.0 – 0.1 | Nulo / No entregado | Vacío, totalmente erróneo o fuera de tema. |

---

## 2. E1 — Pipeline vocab + padding p95 + DataLoader (vale 1.0)

**Variables obligatorias:** `vocab_size` (int), `pad_len` (int), `X_train_shape` (tuple), `y_train_mean` (float).

**Pesos internos (suman 1.0):**
- 0.4 — Vocab correcto (`[PAD]=0, [UNK]=1`, frecuencia >1, tokenización minúsculas + `re.sub(r"[^\w\s]","")`).
- 0.3 — `pad_len` = percentil 95 calculado **solo en train** (aceptado 28–35) + `encode_and_pad` trunca/padea y mapea OOV a `[UNK]`.
- 0.2 — Split 80/20 estratificado `random_state=42` + `DataLoader(batch=64)`, shuffle solo en train. `X_train_shape[0] ≈ 4459`, `0.10 <= y_train_mean <= 0.17`.
- 0.1 — Reporta desbalance en palabras (ej. ~13% spam).

| Nota Ei | Criterio específico E1 |
|---|---|
| 1.0 | 4 variables existen y pasan los 3 chequeos; split y padding solo con train. |
| 0.9 | Funciona pero un detalle menor (ej. no reporta `y_train_mean` en texto aunque la variable existe). |
| 0.6–0.7 | Vocab sin `[UNK]`/sin filtro >1, o `pad_len` = max absoluto, pero el resto funciona. |
| 0.2–0.5 | `pad_len`/vocab calculados sobre todo el DF (fuga), o `DataLoader` mal armado, o no corre. |
| 0.0–0.1 | Vacío o variables inexistentes. |

**Penalizaciones E1:** fuga train/test confirmada: máximo 0.6. `for i in range(len(df))` con `iloc`: −0.1.

---

## 3. E2 — Recurrentes RNN/LSTM/GRU (vale 1.0)

**Variables obligatorias:** `rnn_acc`, `lstm_acc`, `gru_acc` (floats), `e2_train_losses` (dict `{"rnn":[…5], "lstm":[…5], "gru":[…5]}`).
**Protocolo:** `embedding_dim=64, hidden_dim=128`, `BCEWithLogitsLoss`, `Adam(lr=1e-3)`, 5 épocas, loop manual, eval con `torch.no_grad()` + `sigmoid>0.5`.

**Pesos internos (suman 1.0):**
- 0.4 — `GRUModel` implementada con `nn.GRU` **sin** `cell_state` (desempaqueta solo `hidden`), forward correcto.
- 0.3 — Entrenamiento real: 3 listas de 5 pérdidas con tendencia decreciente (última < primera) y `0.70 <= acc <= 1.0`.
- 0.2 — Comparación honesta (esperado `lstm/gru >= rnn − 0.03`; si no se cumple, lo explica en vez de ocultarlo).
- 0.1 — `predict`/ejemplo de inferencia o tabla comparativa.

| Nota Ei | Criterio específico E2 |
|---|---|
| 1.0 | GRU sin `cell_state` + 3 entrenamientos con pérdidas decrecientes + accs en rango + comparación. |
| 0.9 | Todo funciona pero GRU copiada de LSTM con `cell_state` muerto (no usado) o falta 1 curva. |
| 0.6–0.7 | Solo 2/3 modelos, o épocas distintas por modelo sin justificar, o acc fuera de rango por bug. |
| 0.2–0.5 | No corre, o `BCELoss` con logits sin sigmoide (doble activación), o accs inventados sin `e2_train_losses`. |
| 0.0–0.1 | Vacío o variables inexistentes. |

**Penalizaciones E2:** entrenar sobre test o reportar train_acc como test_acc: máximo 0.5.

---

## 4. E3 — Encoder-only vs. Decoder-only (vale 1.0)

**Variables obligatorias:** `enc_acc`, `enc_auc`, `dec_acc`, `dec_auc` (floats).
**Protocolo:** `d_model=128, heads=4, layers=2–3`, `Adam(lr=1e-4)`, 3 épocas, `accuracy_score` + `roc_auc_score`.

**Pesos internos (suman 1.0):**
- 0.4 — Encoder-only con self-attention bidireccional + pooling promedio **enmascarando `[PAD]`**.
- 0.3 — Decoder-only con **máscara causal triangular** + pooling del último token válido.
- 0.2 — Métricas en rango (`0.80 <= acc <= 1.0`, `0.85 <= auc <= 1.0`) con mismo split que E1/E2.
- 0.1 — Conclusión (esperado `enc >= dec − 0.03` en SMS cortos; si no, lo discute).

| Nota Ei | Criterio específico E3 |
|---|---|
| 1.0 | Ambas máscaras correctas + 4 métricas en rango + conclusión vinculada a máscara/posición. |
| 0.9 | Funciona pero pooling sin máscara explícita (aunque el PAD sea poco frecuente) o conclusión genérica. |
| 0.6–0.7 | Solo un sabor, o máscara causal ausente/invertida, o AUC sin `roc_auc_score`. |
| 0.2–0.5 | No corre, o usa `argmax` sobre logits sin sigmoide, o métricas fuera de rango por bug. |
| 0.0–0.1 | Vacío o variables inexistentes. |

**Penalizaciones E3:** promediar PAD sin máscara y no mencionarlo: −0.2. Capas/épocas muy distintas entre sabores sin justificar: −0.1.

---

## 5. E4 — Robustez: ablación PE + orden + longitud (vale 1.0)

**Variables obligatorias:** `enc_nopos_acc` (float), `acc_shuffled` (float), `acc_drop` (float), `acc_by_length` (dict `{"corto","medio","largo"}`).

**Pesos internos (suman 1.0):**
- 0.3 — Ablación: Encoder **sin** `PositionalEncoding`, mismo resto. Esperado `enc_nopos_acc <= enc_acc + 0.02` + 2 líneas de porqué.
- 0.4 — Test de orden: 200 SMS de test con palabras permutadas (`random.seed(42)`), `acc_drop = acc_orig − acc_shuffled` con `acc_drop > 0.02`.
- 0.3 — Buckets `corto(<=15)/medio(16..pad_len)/largo(>pad_len)`: las 3 claves existen; esperado `largo <= corto + 0.05` (truncado p95 penaliza).

| Nota Ei | Criterio específico E4 |
|---|---|
| 1.0 | 4 variables + 3 chequeos se cumplen + explicación causal (orden/posición/truncado). |
| 0.9 | Chequeos ok pero explicación descriptiva (“baja/sube”) sin mecanismo. |
| 0.6–0.7 | Falta 1 sub-experimento o usa otra muestra/semilla sin declararlo. |
| 0.2–0.5 | `acc_drop <= 0` y lo presenta como éxito, o buckets mal definidos, o no corre. |
| 0.0–0.1 | Vacío o variables inexistentes. |

**Penalizaciones E4:** permutar también las etiquetas o evaluar shuffle en train: máximo 0.5.

---

## 6. E5 — XAI con atención + decisión (vale 1.0)

**Variables obligatorias:** `attn_weights_shape` (tuple), `top_tokens` (dict 4 entradas), `decision_text` (str ≥ 60 palabras).

**Pesos internos (suman 1.0):**
- 0.4 — Atención real: última capa, promedio de cabezas, con máscara de `[PAD]` (≈ 0 sobre PAD). `attn_weights_shape` coherente con `pad_len` (ej. `(4,L,L)` o `(4,heads,L,L)`).
- 0.2 — `top_tokens` con 4 entradas no vacías (3 tokens c/u) + 1 heatmap tokens×tokens.
- 0.4 — `decision_text` 5–8 líneas que elige modelo y lo justifica con ≥2 de: orden vs. BoW (E4), máscara/padding (E3), overfitting/curvas, costo `#params`/latencia.

| Nota Ei | Criterio específico E5 |
|---|---|
| 1.0 | Atención enmascarada + heatmap + top_tokens + decisión causal con ≥2 criterios. |
| 0.9 | Todo ok pero decisión solo descriptiva (“el mejor accuracy”) sin costo/errores. |
| 0.6–0.7 | heatmap sin máscara de PAD o `top_tokens` incompleto, pero hay análisis. |
| 0.2–0.5 | Atención de otra capa/modelo sin declarar, o `decision_text` < 60 palabras o genérico. |
| 0.0–0.1 | Vacío o variables inexistentes. |

---

## 7. Prompt copy-paste para la IA evaluadora

> Copia desde "PROMPT:" hasta "FIN PROMPT" junto con el notebook del estudiante. La IA debe devolver SOLO el JSON, sin texto extra.

**PROMPT:**
```
Actúa como profesor de PLN (UNAL) calificando el Taller 1 con la rúbrica adjunta.
Escala: cada ejercicio E1..E5 vale de 0.0 a 1.0. Nota final 0.0-5.0 = SUMA (no promedio). Aprueba con final >= 3.0.
Entradas: 1) Esta rúbrica, 2) Notebook del estudiante (JSON .ipynb: lee TODAS las celdas code+markdown y el namespace final).
Reglas:
- No inventes variables. Si una variable obligatoria no existe, su chequeo vale 0. Cita evidencia (máx 2 líneas de código o el valor reportado).
- Acepta equivalencias de estilo (naming, apply/.str, dict/tupla) si el output es correcto. Solo baja por fallo funcional o conceptual.
- 1.0 = protocolo + output en rango + explicación causal. 0.9 = output correcto pero explicación solo descriptiva.
- Si el código no corre, máximo 0.5 en ese punto. Indica el error.
- Fuga train/test (vocab/pad_len con test): E1 máximo 0.6 y anótalo en observaciones.
- Redondea cada Ei a 1 decimal y final a 1 decimal. final = round(E1+E2+E3+E4+E5, 1).
Devuelve SOLO este JSON válido:
{
  "email": "",
  "fullName": "",
  "submissionId": "",
  "curso": "PLN UNAL Manizales - Taller 1",
  "escala": "Cada punto 0.0-1.0, final 0.0-5.0 por suma, aprueba >=3.0",
  "ejercicios": [
    {"id":"E1","nota":0.0,"evidencia":"...","justificacion":"...","no_entregado":false},
    {"id":"E2","nota":0.0,"evidencia":"...","justificacion":"...","no_entregado":false},
    {"id":"E3","nota":0.0,"evidencia":"...","justificacion":"...","no_entregado":false},
    {"id":"E4","nota":0.0,"evidencia":"...","justificacion":"...","no_entregado":false},
    {"id":"E5","nota":0.0,"evidencia":"...","justificacion":"...","no_entregado":false}
  ],
  "nota_final": 0.0,
  "aprobado": false,
  "observaciones": "copia/fuga/no_corre/faltantes si aplica, sino ''",
  "retroalimentacion_estudiante": "- Bien: ...\n- Corregir: ...\n- Ver: Clase #..."
}
FIN PROMPT
```

---

## 8. Checklist rápido para la IA (no saltar)

1. ¿E1 tiene las 4 variables y `28<=pad_len<=35`, `3500<=vocab_size<=9500`? ¿Split 80/20 seed 42 solo con train?
2. ¿E2 tiene GRU sin `cell_state` + 3 listas de 5 pérdidas decrecientes + `0.70<=acc<=1.0`?
3. ¿E3 tiene máscara de padding en encoder y causal triangular en decoder + `acc>=0.80, auc>=0.85`?
4. ¿E4 tiene `acc_drop>0.02`, 3 buckets y `enc_nopos_acc<=enc_acc+0.02`?
5. ¿E5 tiene `attn_weights_shape` coherente, PAD≈0, `top_tokens` con 4 entradas y `decision_text` ≥60 palabras?
6. Recalcula `nota_final = E1+E2+E3+E4+E5`. 1.0 solo con causal + output; 0.9 si descriptivo sin mecanismo.

---

## 9. Flujo MCP google-classroom (la IA calificadora ejecuta esto)

En Classroom el coursework del Taller 1 contiene **solo instrucciones + `notebook.ipynb`** (`maxPoints: 5`). Todo lo de esta rúbrica vive en este `.md` local.

Orden de herramientas (nombres exactos del MCP):

1. `classroom_list_assignments({courseId, courseWorkStates: ["PUBLISHED"]})` → ubica el `courseWorkId` del Taller 1 por título. Guarda `maxPoints` (debe ser 5; la nota 0.0–5.0 se sube directa).
2. `classroom_list_students({courseId})` → identidad oficial (`email`, `fullName`, `userId`). No confíes solo en el nombre del archivo.
3. `classroom_list_submissions({courseId, courseWorkId, states: ["TURNED_IN"]})` → por cada entrega toma `submissionId` (`id`), `state`, `updateTime` (tardanza) y adjuntos. Sin entrega (`CREATED` sin adjunto): califica 0.0 con `no_entregado: true` en los 5.
4. Por cada adjunto Drive: `drive_get_file({fileId})` (verifica `.ipynb`) → `drive_download_file({fileId, savePath, returnBase64: false})` a `Taller 1/Entregas/`. Si el adjunto es link de Colab (`linkUrl`), intenta descargar; si no es accesible, califica lo visible y anótalo en `observaciones` (no inventes código).
5. Califica cada `local_file` con el prompt de §7. Opcional: `nbconvert --execute` previo para verificar que corre; si no corre, aplica tope 0.5 por ejercicio afectado.
6. `classroom_grade_submission({courseId, courseWorkId, id: submissionId, grade: nota_final})` con `nota_final` 0.0–5.0 a 1 decimal. No uses `draft: true` salvo que el profesor pida revisión previa.
7. `classroom_return_submission({courseId, courseWorkId, id: submissionId})` para devolver la nota. Adjunta `retroalimentacion_estudiante` como comentario si el flujo lo permite; si no, guárdala en `Taller 1/Notas.csv` (`email,fullName,submissionId,E1,E2,E3,E4,E5,nota_final,aprobado,observaciones`).
8. Política de tardanza (define antes de calificar, ej. −0.5 si `late`): aplícala DESPUÉS del redondeo, sin bajar de 0.0, y regístrala en `observaciones`.

*Fin rúbrica Taller 1 — Profesor: Lucas Iturriago — Cada punto vale 1.0, final por suma.*

# T1 · Estructura y contrato base

Verificación local del 25 de noviembre de 2026 en `grupo5/estructura-sdd`, basada en `origin/main` (`a07107c`).

| Comprobación | Resultado observado |
| --- | --- |
| Git y estructura | Archivos exigidos presentes y con contenido. Cambios dentro de `Proyecto-Final-Grupo5/`. `.env` y `app/.streamlit/secrets.toml` ignorados. `.env.example` disponible para versionar. `git diff --check` pasó. |
| Cliente en Python 3.12 | `python -m py_compile app/app.py` pasó. Con `streamlit.testing.v1.AppTest` y respuestas HTTP simuladas, pasó el estado inicial, dos turnos con la misma sesión, la evidencia JSON, el intento sin marca con cero POST de escritura, la confirmación con el borrador esperado, la descarga PDF, la sesión nueva y los errores HTTP 503 y timeout. |
| Validaciones del workflow | Node.js 22 ejecutó los scripts de los nodos Code del export con ocho entradas válidas o inválidas. Pasaron las validaciones de sesión, mensaje, borrador y confirmación booleana. La base responde `503` para entradas válidas mientras se implementan las dependencias. |
| Webhooks publicados en una instancia local de prueba | n8n 2.19.0 importó el export con `n8n import:workflow --input=n8n/workflow.json` y lo publicó con `n8n publish:workflow --id=JalecoG5Base2026`. `POST /webhook/jaleco-chat` respondió `400 INVALID_INPUT` y `503 AGENT_UNAVAILABLE`. `POST /webhook/jaleco-radicar` respondió `400 CONFIRMATION_REQUIRED` sin marca y `503 PERSISTENCE_UNAVAILABLE` con confirmación válida. Los cuatro cuerpos fueron JSON y contenían `output`. |
| Fuente de la presentación | `pdflatex Presentacion.tex` compiló. `pdfinfo` reportó diez páginas de relación 16:9. Se revisó visualmente la diapositiva de arquitectura. |

Estas comprobaciones cubren el contrato inicial y la interfaz con respuestas simuladas. Los criterios de agente, políticas y persistencia requieren ejecución integrada en las tareas T2–T6.

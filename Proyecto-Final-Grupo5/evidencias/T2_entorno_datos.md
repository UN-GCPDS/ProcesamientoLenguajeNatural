# T2 · Entorno reproducible y datos

Verificación local del 25 de noviembre de 2026 en `grupo5/estructura-sdd`. Se ejecutó Docker Engine 29.10.1 y Compose v5.8.0 en Ubuntu 26.04 bajo WSL 2. Los contenedores ejecutaron n8n 2.49.6, PostgreSQL 17.12 y Python 3.12.16 con Streamlit 1.67.0. `python -m pip check` pasó en la imagen de Streamlit.

## Resultados observados

| Prueba | Resultado |
| --- | --- |
| Arranque | `docker compose config --quiet` y `docker compose up -d --build --wait` completaron. Los tres servicios reportaron `healthy`. Streamlit y el editor respondieron en loopback. |
| Carga y permisos | PostgreSQL tenía cuatro pedidos, cuatro políticas y un festivo. `jaleco_reader` tiene `SELECT` y carece de `INSERT`, `UPDATE`, `DELETE` y `TRUNCATE` en las tres tablas. |
| Pedido conocido | `200`. La ejecución n8n 14 ejecutó el nodo de PostgreSQL. La ejecución 26 se inició desde Streamlit en Edge y el panel mostró la respuesta JSON y el valor de `deliveredOn`. |
| Pedido inexistente | `200` con solicitud de corrección. La ejecución 15 consultó PostgreSQL y la ejecución 27 reprodujo el resultado en Streamlit. |
| Identificación | Las ejecuciones 16 y 17 devolvieron aclaración con `200` y carecen de ejecución del nodo PostgreSQL. La ejecución 18 normalizó el ID en minúsculas y consultó una sola vez. |
| Entrada y autenticación | Cuerpo incompleto y sesión inválida devolvieron `400 INVALID_INPUT`. Una clave ausente o incorrecta devolvió `403` antes de crear una ejecución. La ruta de radicación también exigió la clave y mantuvo `400 CONFIRMATION_REQUIRED` o `503 PERSISTENCE_UNAVAILABLE`, según la confirmación. |
| Fallo y recuperación | Con PostgreSQL detenido, la consulta devolvió `503 DATA_SOURCE_UNAVAILABLE`. Al iniciarlo, la misma consulta volvió a devolver `200` con el mismo registro. |
| Repetición desde estado limpio | Una copia de los archivos versionables se inició como `jaleco-t2-check` con puertos alternativos, un `.env` nuevo y volúmenes vacíos. Las dos instancias se configuraron mediante la API local del editor a partir del export sin credenciales del repo. En la segunda pila, las ejecuciones 20 y 21 reprodujeron las respuestas en Streamlit. Los recuentos y permisos coincidieron. |

## Respuestas HTTP crudas del chat

Pedido `JS-101`, HTTP `200`.

```json
{"output":"El pedido JS-101 corresponde a Balón de entrenamiento y se entregó el 2026-11-13.","evidence":[{"source":"pedidos","recordId":"JS-101","data":{"orderId":"JS-101","sku":"BAL-01","productName":"Balón de entrenamiento","channel":"online","deliveredOn":"2026-11-13","warrantyMonths":12,"requiresSerial":false}}],"draft":null}
```

Pedido `JS-999`, HTTP `200`.

```json
{"output":"No encontré ese pedido. Verifica el número e inténtalo de nuevo.","evidence":[],"draft":null}
```

Consulta con PostgreSQL detenido, HTTP `503`.

```json
{"output":"No se pudo consultar la fuente de pedidos. Inténtalo de nuevo en unos minutos.","error":{"code":"DATA_SOURCE_UNAVAILABLE"}}
```

El resultado verifica el grounding de CA-03 y la preparación reproducible de CA-08. El modelo, la memoria, la evaluación de políticas, la radicación y el ensayo de la demo corresponden a T3–T6.

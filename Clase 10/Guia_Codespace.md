# Clase 10 — Agente UNAL con n8n en GitHub Codespaces

Sin cuenta de n8n Cloud. n8n corre con Docker dentro del Codespace.

## 1. Abrir el Codespace (estudiantes, 2 min)

1. GitHub > `Code` > `Codespaces` > `Create codespace on main`.
2. Esperar al `postCreateCommand` (`pip install requests ...`).
3. Verificar puertos: pestaña `PORTS` debe mostrar `5678 - n8n`.

> El repo monta `..:/home/node/files:ro` en el contenedor n8n, por eso el workflow ya apunta a `/home/node/files/Clase 10/Inventario Laboratorio UNAL.xlsx`. No hay que subir el Excel a mano.

## 2. Levantar n8n

```bash
cd .devcontainer
docker compose up -d
docker compose ps
```

Abrir `http://localhost:5678` (o la URL forwardeada `https://TU-CODESPACE-5678.app.github.dev`). Primera vez: crear usuario local (admin / password que quieras, es solo local).

## 3. Importar el workflow

1. n8n > `Workflows` > `...` > `Import from file` > selecciona `Clase 10/n8n-workflow.json`.
2. Verás: `Webhook` → `AI Agent` + `OpenRouter Chat Model` + `Window Buffer Memory` + tools `Inventario_UNAL` y `Clima_Manizales`.

## 4. Credencial OpenRouter (único paso con key externa)

1. En el nodo `OpenRouter Chat Model` > `Credential` > `Create new`.
2. Pega tu key de [OpenRouter](https://openrouter.ai/keys) (gratuita, con saldo free).
3. Modelo default `meta-llama/llama-3.3-70b-instruct:free`, temperatura `0` (determinismo para invocar tools, ver slides). Puedes cambiar a cualquier otro modelo `:free`.

## 5. Activar y probar

1. Arriba a la derecha: pasa el workflow a `Active`.
2. Usa la **Production URL**: `.../webhook/agente-logistica-unal` (NO `webhook-test`, esa solo escucha con `Execute workflow` abierto).

```bash
# dentro del Codespace (dev container ve a n8n por localhost:5678)
python "Clase 10/Peticion.py"
python "Clase 10/Peticion.py" --mensaje "¿Hay osciloscopios?" --session-id lucas-01
```

Payload que espera el Webhook (mapeado en el Agent y en Memory):

```json
{"sessionId": "lucas-unal-001", "chatInput": "¿...?"}
```

* `sessionId` → `Window Buffer Memory` (`={{ $('Webhook').item.json.body.sessionId }}`), resuelve el error `No Session ID found`.
* `chatInput` → `AI Agent` (`={{ $json.body.chatInput }}`).

Si quieres apuntar a la URL pública forwardeada:

```bash
export N8N_WEBHOOK_URL="https://TU-CODESPACE-5678.app.github.dev/webhook/agente-logistica-unal"
python "Clase 10/Peticion.py"
```

## 6. Observar (MLOps)

Pestaña `Executions` en n8n: inspecciona el JSON por arista, tokens/min y latencia. Es la evidencia del ReAct (Thought → Action → Observation).

## 7. Apagar

```bash
cd .devcontainer
docker compose stop n8n   # pausa, conserva n8n_data
# docker compose down -v   # SOLO si quieres borrar credenciales y empezar de cero
```

El Codespace se suspende a los 30 min sin uso. Al retomarlo, repite `docker compose up -d`.

---

## Solución de problemas

**`docker: command not found`:**
Las versiones viejas del Codespace no traían el CLI de Docker. Ya está corregido en `.devcontainer/devcontainer.json` (feature `docker-outside-of-docker`). Aplica con: `Ctrl+Shift+P` → `Codespaces: Rebuild Container`.

**Matar todo y arrancar desde cero:**
```bash
cd .devcontainer
docker compose down        # apaga y borra contenedores, CONSERVA datos de n8n
docker compose down -v     # lo anterior + BORRA n8n_data (usuarios, workflows,
                           # credenciales: n8n queda de fabrica, hay que reimportar todo)
docker compose pull n8n
docker compose up -d
```
Opción nuclear (VM nueva, cero absoluto): borra el Codespace en `github.com/codespaces` y crea uno nuevo con `Create codespace on main`.

**Me pide actualizar n8n / tengo versión vieja:**
```bash
cd .devcontainer
docker compose pull n8n
docker compose up -d --force-recreate n8n
```
(El compose ya usa `n8nio/n8n:latest`. La migración de la base interna es automática; tus workflows y credenciales están en el volumen `n8n_data` y se conservan.)

**Banner "Connection lost" todo el tiempo:**
Causa: el editor de n8n pierde el canal de eventos porque genera sus URLs con `localhost`, pero tú lo abres por la URL forwardeada del proxy de Codespaces.
1. Abre n8n **siempre** con la URL https de la pestaña `PORTS` (`https://TU-CODESPACE-5678.app.github.dev`), nunca con `http://localhost:5678`.
2. Si persiste: copia esa URL, descomenta `N8N_EDITOR_BASE_URL` y `WEBHOOK_URL` en `.devcontainer/docker-compose.yml` con tu URL (con `/` final) y recrea:
```bash
cd .devcontainer
docker compose up -d --force-recreate n8n
```
3. Recarga el navegador con caché limpia (Ctrl+Shift+R). Verifica que el puerto 5678 esté en `Private` o `Public` en `PORTS` (con `Private` basta si lo abres en el mismo navegador donde logueaste GitHub).

---

## Apéndice para el docente — demo en vivo: Google Sheets

El workflow por defecto usa `Spreadsheet File` local para evitar OAuth en clase. Para enseñar el grounding con Sheets sin romper nada:

1. Duplica el workflow (por seguridad).
2. Añade nodo `Google Sheets Tool` > `Resource: Sheet within document` > `Operation: Get Rows`, autentica con OAuth y elige el doc `Inventario Laboratorio UNAL`.
3. Copia el `toolDescription` del nodo `Inventario_UNAL` al nuevo nodo.
4. `Disable` el nodo `Inventario_UNAL`, conecta el Sheets como `ai_tool` al `AI Agent`.
5. `Execute workflow` con `{"sessionId":"demo-01","chatInput":"¿Hay Raspberry Pi?"}` y muestra que el plan ReAct es idéntico.

El Sticky Note dentro del workflow resume estos pasos.

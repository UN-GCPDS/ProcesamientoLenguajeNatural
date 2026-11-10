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

1. n8n > `Workflows` > `...` > `Import from file` > selecciona `Clase 10/Clase10 - Agente Logistica UNAL.json`.
2. Verás: `Webhook` → `AI Agent` + `OpenRouter Chat Model` + `Window Buffer Memory` + tools `Inventario_UNAL` y `Clima_Manizales`.

## 4. Credencial OpenRouter (único paso con key externa)

1. En el nodo `OpenRouter Chat Model` > `Credential` > `Create new`.
2. Pega tu key de [OpenRouter](https://openrouter.ai/keys) (gratuita, con saldo free).
3. Modelo verificado en clase `nvidia/nemotron-3.5-super-vl:free`, temperatura `0` (determinismo para invocar tools, ver slides). Puedes cambiar a cualquier otro modelo `:free` (ej. `meta-llama/llama-3.3-70b-instruct:free`).

## 4b. Conexión Google Sheets con OAuth2 en Codespaces

> Resumen operativo (OAuth2 en Codespaces):

1. `PORTS > 5678 > Port Visibility > Public` (obligatorio, si queda en `Private` Google bloquea el login y el callback no vuelve).
2. En n8n: `Credentials > Add Credential > Google Sheets OAuth2 API`, copia el `OAuth Redirect URL` (`https://TU-CODESPACE-5678.app.github.dev/rest/oauth2-credential/callback`).
3. En [Google Cloud Console](https://console.cloud.google.com/) (proyecto `DesarrolloClases` o nuevo): `APIs y servicios > Biblioteca`, habilita `Google Sheets API` + `Google Drive API` (Drive es obligatoria para listar por nombre).
4. `Google Auth Platform > Información de la marca` (app `n8n`, tus correos) > `Público > Externo` + `Add test users > tu correo` (si lo omites: `Error 403: access_denied`) > `Acceso a los datos` (scopes de Sheets/Drive).
5. `Clientes > Crear cliente > Aplicación web > Authorized redirect URIs > + Agregar URI`, pega exacta la URL del paso 2 (letra por letra, si no: `redirect_uri_mismatch`).
6. Copia `Client ID` + `Client Secret` a n8n. **No pulses `Save` todavía**: primero `Sign in with Google` (elige la cuenta test user, `Advanced > Ir a n8n (no seguro)` si sale aviso, acepta permisos), espera el banner verde `Account connected` y **ahora sí `Save`**. Si guardas antes: `Request failed with status code 404`.
7. Sube `Inventario Laboratorio UNAL` a Drive con columnas `ID,Componente,Categoria,Cantidad,Ubicacion,Estado`. En el nodo `Google Sheets Tool`: `Resource: Sheet within document`, `Operation: Get Row(s)`, `Document: Inventario...`, `Sheet: Hoja 1`, sin `Filters`, `Tool Description` igual al del nodo local. Haz `Disable` al `Spreadsheet File`.

Si recreas el Codespace la URL cambia: agrega el nuevo redirect URI en GCP.

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

**`Status 401` con body vacío al usar `Peticion.py` con la URL `https://...`:**
No es n8n, es el proxy del Codespace. Corres el script **dentro** del Codespace contra la URL pública con el puerto en `Private`: GitHub exige login y Python no lo lleva (`Executions` no registra nada).
1. Usa `http://localhost:5678/webhook/agente-logistica-unal` desde dentro del Codespace, o
2. Pon `PORTS > 5678 > Port Visibility > Public` y reintenta la URL `https://...`.

**Aviso `WEBHOOK_URL -> Use N8N_WEBHOOK_URL instead` (n8n 2.x):**
Funciona con `WEBHOOK_URL`, pero en v2 está deprecada. Cambia a `N8N_WEBHOOK_URL` en `docker-compose.yml` para silenciarlo.

**OAuth Google: `404` al guardar, `403 access_denied`, `redirect_uri_mismatch` o callback en blanco:**
No guardar antes de `Sign in`, agregar test user, copiar redirect URI exacta, puerto `5678` en `Public`.

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

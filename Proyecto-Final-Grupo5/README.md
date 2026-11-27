# Jaleco Sports · Agente de postventa

Proyecto final de Procesamiento de Lenguaje Natural · Grupo 5 · UNAL Manizales.

Jaleco Sports es una tienda ficticia de artículos deportivos. El agente transforma un reclamo redactado en español en una solicitud estructurada, consulta el pedido, explica la regla aplicable y permite radicar el caso con confirmación explícita.

## Archivos de la entrega

```text
Proyecto-Final-Grupo5/
├── README.md
├── SPEC.md
├── PLAN.md
├── TASKS.md
├── AGENTS.md
├── Presentacion.pdf
├── Presentacion.tex
├── compose.yaml
├── .env.example
├── n8n/
│   ├── workflow.json
│   └── db/init.sql
├── app/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── skills/
│   ├── reglas-postventa/SKILL.md
│   └── contrato-radicacion/SKILL.md
└── evidencias/
```

[SPEC.md](SPEC.md) define qué se evalúa. [PLAN.md](PLAN.md) fija las interfaces y el diseño. [TASKS.md](TASKS.md) registra qué se ha verificado y qué falta.

## Ejecutar T2 en local

Use [Docker Engine para Ubuntu](https://docs.docker.com/engine/install/ubuntu/) con el complemento Compose. En Windows, ejecute estos pasos desde Ubuntu en WSL y mantenga abierta esa terminal mientras use los servicios. Compruebe el daemon con `sudo docker version` y el complemento con `sudo docker compose version`.

1. Desde `Proyecto-Final-Grupo5/`, copie `.env.example` como `.env`. Genere un valor diferente para cada secreto con `openssl rand -hex 24`. Complete `POSTGRES_PASSWORD`, `JALECO_READER_PASSWORD` y `N8N_WEBHOOK_KEY`. El `.env` se queda en su máquina.
2. Valide y levante la pila.

   ```sh
   sudo docker compose config --quiet
   sudo docker compose up -d --build --wait
   sudo docker compose ps
   ```

3. Abra el editor local de n8n en `http://localhost:5678`. Si es la primera vez, cree la cuenta de propietario e importe `n8n/workflow.json` desde **Import from File**.
4. En el nodo **Consultar pedido en PostgreSQL**, cree una credencial PostgreSQL con host `postgres`, puerto `5432`, base de datos `jaleco`, usuario `jaleco_reader`, contraseña `JALECO_READER_PASSWORD` y SSL deshabilitado para la red local. Use la prueba de conexión del editor.
5. En **Webhook chat**, cree una credencial **Header Auth** cuyo nombre de cabecera sea `X-Jaleco-Key` y cuyo valor sea `N8N_WEBHOOK_KEY`. Seleccione la misma credencial en **Webhook radicación**. Guarde y publique el workflow.
6. Abra Streamlit en `http://localhost:8501`. Envíe «Consulta el pedido JS-101» y abra **Evidencia · respuesta cruda**. Debe mostrar `recordId: "JS-101"` y `deliveredOn: "2026-11-13"`. Consulte `JS-999` y compruebe que pide corregir el número. T2 consulta pedidos. La interpretación del reclamo y la radicación se incorporan en las fases siguientes.

La carga SQL se ejecuta al inicializar un volumen de PostgreSQL vacío. `sudo docker compose stop` detiene los servicios y conserva sus volúmenes. Para repetir desde cero sin afectar sus datos, cree otro `.env` con secretos nuevos y puertos libres, por ejemplo `N8N_HOST_PORT=15678` y `STREAMLIT_HOST_PORT=18501`. Desde una copia limpia del proyecto ejecute `sudo docker compose -p jaleco-t2-check up -d --build --wait` y repita la configuración de n8n en el nuevo puerto.

Para trabajar únicamente en la interfaz, instale `app/requirements.txt` en un entorno virtual y ejecute `python -m streamlit run app/app.py` desde esta carpeta. En ese caso use `app/.env.example` como plantilla y ajuste `app/.env` con `N8N_BASE_URL=http://localhost:5678` y la clave del webhook local. Con Compose, la aplicación recibe la URL interna y la clave desde su servicio.

## Demostración prevista

Describir un reclamo, completar el pedido en un segundo turno y comprobar que una sesión nueva requiere identificarlo de nuevo. Mostrar la evidencia del pedido y la evaluación del plazo, intentar radicar sin marcar la casilla y confirmar. Verificar el registro y descargar su constancia.

El [material de las Clases 10 y 11](https://github.com/UN-GCPDS/ProcesamientoLenguajeNatural) explica el patrón de referencia. Las [evidencias T0](evidencias/) corresponden al arranque del entorno del curso. La [presentación inicial](Presentacion.pdf) distingue ese avance de la evaluación pendiente.

Para regenerar la presentación desde esta carpeta con LaTeX instalado, ejecute `pdflatex Presentacion.tex`.

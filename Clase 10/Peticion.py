"""
Clase 10 - Agente de logistica UNAL (n8n self-hosted en Codespaces).

Uso basico (dentro del Codespace, n8n en localhost:5678):
    python Peticion.py
    python Peticion.py --mensaje "¿Hay osciloscopios?" --session-id lucas-01
    python Peticion.py --url https://TU-CODESPACE-5678.app.github.dev/webhook/agente-logistica-unal

Prioridad de la URL:
    1. --url
    2. env N8N_WEBHOOK_URL
    3. http://localhost:5678/webhook/agente-logistica-unal (Codespaces / docker-compose)

Formato esperado por el workflow (ver n8n-workflow.json):
    {"sessionId": "...", "chatInput": "..."}

Notas n8n self-hosted:
    - Usa la Production URL (/webhook/...) con el workflow en Active.
    - La Test URL (/webhook-test/...) solo escucha mientras das "Execute workflow" en la UI.
"""

import argparse
import json
import os
import sys

try:
    import requests
except ImportError:
    print("Falta 'requests'. Instala con: pip install requests", file=sys.stderr)
    sys.exit(1)


DEFAULT_LOCAL_URL = "http://localhost:5678/webhook/agente-logistica-unal"
# URL historica en n8n Cloud (referencia, ya no es el default):
# https://liturriago.app.n8n.cloud/webhook-test/agente-logistica-unal


def parse_args():
    p = argparse.ArgumentParser(description="Invoca al agente de logistica UNAL en n8n.")
    p.add_argument("--url", default=os.getenv("N8N_WEBHOOK_URL", DEFAULT_LOCAL_URL),
                   help="Webhook de n8n (default: env N8N_WEBHOOK_URL o localhost)")
    p.add_argument("--session-id", default="lucas-unal-001",
                   help="ID de sesion para Window Buffer Memory")
    p.add_argument("--mensaje", default="¿Me confirmas si la Raspberry Pi 4 Model B 4GB de los laboratorios de la UNAL está disponible?",
                   help="Pregunta al agente")
    p.add_argument("--timeout", type=int, default=120, help="Timeout en segundos")
    return p.parse_args()


def main():
    args = parse_args()
    payload = {"sessionId": args.session_id, "chatInput": args.mensaje}

    print(f"POST {args.url}")
    print(f"Payload: {json.dumps(payload, ensure_ascii=False)}")

    try:
        response = requests.post(args.url, json=payload, timeout=args.timeout)
    except requests.exceptions.ConnectionError:
        print("\nERROR: no se pudo conectar a n8n.", file=sys.stderr)
        print("Verifica que: 1) el Codespace esta corriendo, 2) docker compose up -d, 3) n8n en http://localhost:5678", file=sys.stderr)
        sys.exit(1)

    print(f"Status: {response.status_code}")
    if response.status_code == 404:
        print("404: revisa que el path sea /webhook/agente-logistica-unal y que el workflow este en Active (no uses /webhook-test/ para produccion).", file=sys.stderr)

    try:
        data = response.json()
    except ValueError:
        print("Respuesta no-JSON:")
        print(response.text)
        return

    # n8n con responseMode=whenLastNodeFinishes suele devolver [{"output": "..."}] o {"output": "..."}
    if isinstance(data, list) and data and isinstance(data[0], dict) and "output" in data[0]:
        print("\nRespuesta del agente:\n" + str(data[0]["output"]))
    elif isinstance(data, dict) and "output" in data:
        print("\nRespuesta del agente:\n" + str(data["output"]))
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

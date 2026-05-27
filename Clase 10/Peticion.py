import requests

WEBHOOK_URL = "https://liturriago.app.n8n.cloud/webhook-test/agente-logistica-unal"

# Estructura del payload JSON corregida con el Session ID requerido
payload = {
    "sessionId": "lucas-unal-001",
    "chatInput": "¿Me confirmas si la Raspberry Pi 4 Model B 4GB de los laboratorios de la UNAL está disponible?"
}

print("Enviando petición con Session ID al agente...")
response = requests.post(WEBHOOK_URL, json=payload)
print(response.json())
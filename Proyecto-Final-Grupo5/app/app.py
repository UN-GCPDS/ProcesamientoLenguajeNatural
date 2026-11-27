import os
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF


load_dotenv(Path(__file__).with_name(".env"))
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "http://n8n:5678").rstrip("/")
N8N_WEBHOOK_KEY = os.getenv("N8N_WEBHOOK_KEY", "")

CATEGORIES = {
    "cambio_talla": "Cambio por talla",
    "defecto": "Producto defectuoso",
    "pedido_incompleto": "Pedido incompleto",
    "retracto": "Retracto",
    "sin_clasificar": "Motivo por confirmar",
}
DECISIONS = {
    "elegible_preliminar": "Elegible preliminarmente",
    "fuera_de_plazo_preliminar": "Fuera de plazo, sujeto a revisión",
    "evidencia_pendiente": "Evidencia pendiente",
    "revision_humana": "Revisión humana",
}


def post_to_n8n(path, body, success_codes):
    headers = {"X-Jaleco-Key": N8N_WEBHOOK_KEY} if N8N_WEBHOOK_KEY else {}
    try:
        response = requests.post(
            f"{N8N_BASE_URL}/webhook/{path}",
            json=body,
            headers=headers,
            timeout=(5, 90),
            allow_redirects=False,
        )
    except requests.RequestException:
        return False, "No se pudo conectar con el agente. Intenta de nuevo.", None

    try:
        raw = response.json()
    except ValueError:
        raw = response.text

    output = raw.get("output") if isinstance(raw, dict) else None
    if response.status_code not in success_codes:
        error = output if isinstance(output, str) and output else None
        return False, error or f"El agente devolvió HTTP {response.status_code}.", raw

    if not isinstance(output, str) or not output.strip():
        return False, "La respuesta del agente tiene un formato inesperado.", raw
    return True, output, raw


def has_text_fields(record, fields):
    return isinstance(record, dict) and all(
        isinstance(record.get(field), str) and record[field].strip() for field in fields
    )


def show_message(message):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["text"])
        else:
            st.write(message["text"])
        if message.get("raw") is not None:
            with st.expander("Evidencia · respuesta cruda"):
                raw = message["raw"]
                if isinstance(raw, (dict, list)):
                    st.json(raw)
                else:
                    st.code(str(raw), language=None)


def receipt_pdf(case):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.multi_cell(0, 10, "Constancia de radicación · Jaleco Sports", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    values = (
        ("Radicado", case["caseId"]),
        ("Pedido", case["orderId"]),
        ("Motivo", CATEGORIES.get(case["category"], case["category"])),
        ("Fecha", case["createdAt"]),
        ("Resultado", DECISIONS.get(case["decision"], case["decision"])),
    )
    for label, value in values:
        line = f"{label}   {value}".encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


st.set_page_config(page_title="Postventa · Jaleco Sports", page_icon="🧾")
st.title("Postventa de Jaleco Sports")

state = st.session_state
if st.sidebar.button("Nueva conversación") or "session_id" not in state:
    state.session_id = str(uuid4())
    state.messages = []
    state.draft = None
    state.case = None
st.sidebar.caption(f"Sesión {state.session_id}")

if not state.messages:
    st.info("Describe lo que ocurrió y escribe el número de pedido si lo tienes.")

for message in state.messages:
    show_message(message)

if prompt := st.chat_input("Describe tu solicitud", max_chars=2000):
    prompt = prompt.strip()
    if not prompt:
        st.warning("Escribe un mensaje antes de enviarlo.")
    else:
        user_message = {"role": "user", "text": prompt}
        state.messages.append(user_message)
        show_message(user_message)

        with st.spinner("Consultando al agente…"):
            success, output, raw = post_to_n8n(
                "jaleco-chat", {"sessionId": state.session_id, "chatInput": prompt}, {200}
            )
        if success:
            draft = raw.get("draft")
            state.draft = draft if has_text_fields(draft, ("draftId", "orderId", "category", "decision")) else None

        reply = {"role": "assistant", "text": output, "raw": raw, "error": not success}
        state.messages.append(reply)
        show_message(reply)

if draft := state.draft:
    with st.form(key=f"radicar-{draft['draftId']}"):
        st.subheader("Revisar solicitud")
        st.write(f"**Pedido** {draft['orderId']}")
        st.write(f"**Motivo** {CATEGORIES.get(draft['category'], draft['category'])}")
        st.write(f"**Resultado** {DECISIONS.get(draft['decision'], draft['decision'])}")
        confirmed = st.checkbox("Confirmo que quiero radicar esta solicitud", key=f"confirmar-{draft['draftId']}")
        submitted = st.form_submit_button("Radicar caso")

    if submitted and not confirmed:
        st.warning("Marca la casilla de confirmación antes de radicar.")
    elif submitted:
        with st.spinner("Radicando el caso…"):
            success, output, raw = post_to_n8n(
                "jaleco-radicar",
                {"sessionId": state.session_id, "draftId": draft["draftId"], "confirmed": True},
                {200, 201},
            )
        case = raw.get("case") if success else None
        if has_text_fields(case, ("caseId", "orderId", "category", "decision", "createdAt")):
            state.case = case
            state.draft = None
            st.rerun()
        st.error("La respuesta no incluyó el radicado completo. Puedes reintentar." if success else output)

if case := state.case:
    st.success(f"Caso radicado · {case['caseId']}")
    st.download_button("Descargar constancia", receipt_pdf(case), "constancia-jaleco.pdf", "application/pdf")

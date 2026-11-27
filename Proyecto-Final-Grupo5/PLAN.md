# PLAN · Architecture and contracts

## Architecture

Streamlit → internal n8n HTTP webhooks → agent, rules, and persistence → PostgreSQL. Streamlit, n8n, and PostgreSQL share the same deployment environment. The browser connects to Streamlit through its published port. Streamlit calls `http://n8n:5678` over the Compose network, and n8n connects to PostgreSQL using its service name.

| Component | Responsibility |
| --- | --- |
| Streamlit | Receive messages, validate format and length, keep visible history for each session, display the received JSON, present the confirmation form, and generate the PDF receipt. If attachments are enabled, the application validates them and extracts their contents. |
| n8n · Chat | Validate the contract, run AI Agent at temperature 0 with memory keyed by `sessionId`, look up source records, and evaluate rules. Produce a draft containing the case fields the user can review. |
| n8n · Case filing | Validate confirmation, session ownership, and draft expiration. Create the case idempotently and return the stored record. |
| PostgreSQL | Store orders, versioned policies, the test calendar, memory, drafts, and cases. A unique constraint on `casos.draft_id` supports idempotency. |

PostgreSQL stores Jaleco data. It supports scripted seeding and can enforce consistency and uniqueness within a transaction when case filing is added. n8n uses SQLite in its own persistent volume for workflow configuration. T2 seeds orders, policies, and a holiday calendar on an empty PostgreSQL volume. The n8n credential has read-only access to those tables. The Google Sheets credentials and the course workflow test document the T0 setup.

The contract retains `sessionId`, `chatInput`, and `output` from the course example. The Respond to Webhook node returns a JSON object with a stable schema. The interface uses that object and records the response body as evidence for the turn.

T2 uses a single Compose network with both browser-facing ports bound to loopback. Streamlit reaches n8n by service name. Local setup requires attaching a Header Auth credential to both webhook nodes and a PostgreSQL credential to the lookup node before publishing.

## Tools and design decisions

| Operation | When to use it and required parameters | Restrictions |
| --- | --- | --- |
| `consultar_pedido` | Given a specific `orderId`, retrieve one row by ID using a parameterized query. | Use synthetic data and authorized fields only. Reject bulk order listings, writes, and SQL supplied by the model. |
| `consultar_politica` | Given the normalized category and effective-date context, retrieve the applicable policy. | Read-only. Do not invent exceptions or substitute text from the chat for the policy rule. |
| `evaluar_solicitud` | Given the retrieved order and policy, calculate the outcome using the calendar, cutoff date, and evidence fields. | Use deterministic code and set test dates outside the chat. The structured result determines the decision returned to the user. |
| `radicar_caso` | Call through the write webhook with `sessionId`, `draftId`, and explicit boolean confirmation. | Reject case creation through free-form chat. Do not accept draft fields modified by the client. |

The LLM interprets the reason, extracts information supplied by the customer, and asks follow-up questions. In the assessment path, n8n validates the category and uses the order ID to retrieve authorized data. The code determines the outcome, and the formal response uses that structured result. The response evidence and the draft use the same result. Each draft stores an immutable snapshot and expires after a limited period. The write path uses the draft ID as the idempotency key and verifies session ownership when it returns an existing case as well.

## HTTP contract

Both endpoints accept `Content-Type: application/json`. `sessionId` is a UUID that stays the same throughout the conversation. Streamlit generates another UUID for each new session. The Streamlit server sends the Header Auth key as `X-Jaleco-Key`. Both published webhooks require this credential.

### Conversation

`POST /webhook/jaleco-chat`

```json
{"sessionId":"9e765fc1-5868-4413-a916-6dc01cda975a","chatInput":"Mi pedido JS-104 llegó incompleto"}
```

`chatInput` accepts 1–2000 characters after trimming whitespace. A successful request returns HTTP `200` with the following body.

```json
{
  "output": "El pedido JS-104 corresponde a Kit de bandas elásticas y se entregó el 2026-11-17.",
  "evidence": [{
    "source": "pedidos",
    "recordId": "JS-104",
    "data": {
      "orderId": "JS-104",
      "sku": "KIT-04",
      "productName": "Kit de bandas elásticas",
      "channel": "online",
      "deliveredOn": "2026-11-17",
      "warrantyMonths": 12,
      "requiresSerial": false
    }
  }],
  "draft": null
}
```

Each item in `evidence` includes `source` and `recordId` and may include the fields used under `data`. When a draft is available, `draft` contains `draftId`, `orderId`, `category`, and `decision`. Expected outcomes are `elegible_preliminar`, `fuera_de_plazo_preliminar`, `evidencia_pendiente`, and `revision_humana`.

In T2, n8n looks for one distinct `JS-###` identifier, normalizes it to uppercase, and performs one parameterized lookup. A message with zero or several distinct IDs receives a clarification with `evidence: []` and `draft: null`. An unknown ID receives a correction prompt with the same fields. Both cases return `200`. T2 always returns `draft: null`.

### Case filing

`POST /webhook/jaleco-radicar`

```json
{"sessionId":"9e765fc1-5868-4413-a916-6dc01cda975a","draftId":"630a57b4-dd5a-491b-901b-67124ea2c474","confirmed":true}
```

The endpoint returns `201` when it creates a case and `200` when it retrieves an existing case for the same draft. Both responses include `output` and `case`. The `case` object contains `caseId`, `orderId`, `category`, `decision`, and an ISO 8601 `createdAt` timestamp.

During T2, the endpoint validates the request and returns `503 PERSISTENCE_UNAVAILABLE` for a valid confirmation. Case creation and retries are implemented in T5.

### Errors

Application error responses include Spanish text in `output` and a diagnostic `error.code`. They use `400` for invalid input or missing explicit confirmation, `404` for an inaccessible or expired unfiled draft once filing is available, and `503` for unavailable dependencies. A lookup failure in T2 returns `503 DATA_SOURCE_UNAVAILABLE`. Native n8n Header Auth returns `403` for a missing or wrong key before workflow execution and may return plain text. The client handles responses outside the application schema.

## Risks and checks

- Date interpretation. Use `America/Bogota`, version the calendar, and test inclusive boundaries. Record the dates observed in the evidence.
- Uncertain classification or ambiguous conditions. Ask for clarification, abstain, or refer the case for human review. Classification alone cannot authorize an action.
- Provider or database failure. Log the failure and return an error in Spanish. A filing retry reuses the same `draftId`.
- Changing dependencies. Pin the verified n8n version and record the model and rule version before the rehearsal. Store secrets in n8n credentials or local environment variables.
- Information exposure. Use synthetic data and omit credentials, personal data, and internal model text from exported evidence.

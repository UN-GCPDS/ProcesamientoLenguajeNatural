---
name: contrato-radicacion
description: Maintain the Streamlit–n8n contract and verify confirmation, session ownership, and idempotency for Jaleco Sports case filing.
---

# API contract and case filing

1. Use the JSON bodies and HTTP status codes in [PLAN.md](../../PLAN.md). The client generates one `sessionId` per conversation and calls the internal URL. The evidence panel shows the response body received in that turn.
2. When the contract changes, exercise the conversation across two sessions and confirm that an invalid request body receives an explanatory error.
3. Show the draft before confirmation. Streamlit sends `sessionId`, `draftId`, and `confirmed: true` only when the user submits the form with the checkbox selected.
4. n8n verifies session ownership and checks whether the draft has expired before persisting its contents. Enforce a unique association by `draftId`. If the case already exists for that session, return the same `caseId`.
5. Compare case counts before and after requesting a write through chat, submitting with the checkbox cleared, using an ID from another session, and submitting valid confirmation. Repeat the valid request sequentially and concurrently. Generate the receipt from the resulting record.

When troubleshooting, distinguish a `404` from the proxy or an unpublished webhook from a draft error identified by `error.code`. Check the execution ID in n8n and omit authentication headers from archived evidence.

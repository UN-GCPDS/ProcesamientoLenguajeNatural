# AGENTS · Jaleco Sports

## Scope of work

- Keep the deliverables in `Proyecto-Final-Grupo5/`. Use the course repository as a reference. Edit course materials only when the task explicitly requires it.
- Read `SPEC.md`, `PLAN.md`, and `TASKS.md` before changing the workflow. Consult `skills/` for after-sales rules and API conventions.
- Update the workflow, Streamlit, and the examples together when the contract changes. Record the rationale for each new decision and update the affected criteria.
- Record the evidence you collected and the scope of each test. An HTTP `200` from the course workflow verifies the course workflow only.

## Running the project

Run Docker Compose from this directory to start PostgreSQL, n8n, and Streamlit. Follow `README.md` to set local credentials, import the workflow, and publish it on the pinned n8n version. For client-only development, install `app/requirements.txt` and run `python -m streamlit run app/app.py` against a configured local webhook.

## Project rules

- Keep language interpretation in the agent and rule-based decisions and persistence in n8n. Use parameters, versioned policies, and reproducible cutoff dates.
- The chat retrieves information and prepares drafts. Case filing validates confirmation and session ownership on the server, returns the stored result, and handles retries.
- Use synthetic data and include only the fields required for evidence. Keep `.env` files, keys, tokens, passwords, and credentials out of Git.
- Check the Git diff and status before closing a task. Push changes when the team authorizes it.

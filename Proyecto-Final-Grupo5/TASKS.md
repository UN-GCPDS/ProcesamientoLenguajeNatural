# TASKS · Group 5 deliverables

For each completed task, record the commit, date, command or test steps, and observed result in `evidencias/`. Acceptance criteria are in [SPEC.md](SPEC.md).

| Status | Task | Criteria | Completion evidence |
| --- | --- | --- | --- |
| Completed in Juan's environment | T0 · Reference environment setup | Prerequisite for CA-08 | [Docker status](evidencias/T0_docker_ps.txt), [HTTP 200](evidencias/T0_status200.txt), and [credentials](evidencias/T0_credenciales.png). The request tested the course workflow. |
| Baseline ready | T1 · Repository structure and contract | CA-01–08 | [Local verification](evidencias/T1_estructura.md) of the delivery tree, baseline contract, Streamlit client, and presentation with its source file. |
| Completed locally | T2 · Reproducible environment and data | CA-03, CA-08 setup | [T2 verification](evidencias/T2_entorno_datos.md) records the pinned versions, seeded data, a live order lookup from Streamlit, and a repeat run on clean volumes. |
| Pending | T3 · Agent and sessions | CA-01–03 | Exported workflow with the model at temperature 0, read tools, and three recorded turns across two sessions. |
| Pending | T4 · Rules and evidence | CA-03, CA-04 | Boundary cases with a fixed date, order and policy JSON, calculated outcomes, and the Streamlit evidence panel. |
| Pending | T5 · Case filing and retries | CA-05–07 | Requests without confirmation and requests made through chat create zero cases. Confirmation creates one row and a PDF generated from that record. Save the results of retry and concurrency tests. |
| Pending | T6 · Evaluation and final presentation | CA-01–08 | Labeled corpus, metrics and errors, comparison of manual and assisted handling, a PDF of up to ten slides, and a rehearsal of up to ten minutes. Record team assignments and add links to the commit and the demo. |

T2 and client development can proceed in parallel using the T1 contract. T3–T5 integrate a withdrawal case that spans a public holiday before expanding coverage. T6 uses evidence collected from the integrated system.

# SPEC · After-sales support agent

## Objective

Help customers and support staff file complete, traceable after-sales requests. Customers can describe the problem in everyday language. Support receives a case containing the order, reason, preliminary outcome, missing information, and a reference to the rule applied.

Handling each claim manually requires interpreting the message, locating the order, looking up the rule, and entering the details. Evaluate the prototype by running the same cases through the manual and assisted processes. Measure time to a usable case record and field completeness. Jaleco Sports is a fictional store, and the demo data is synthetic.

## Scope

- Streamlit is the single support channel. The model distinguishes `cambio_talla`, `defecto`, `pedido_incompleto`, and `retracto`. It may return `sin_clasificar` when asking for clarification or referring the case to support.
- The preliminary assessment looks up the order and applies versioned rules in code. It distinguishes store-policy exchanges, warranty claims, and the right of withdrawal. It uses evidence reported by the customer to guide the next step.
- Creating a case requires explicit confirmation and is the only action that changes business data. Customers can download a case filing receipt.
- Future work includes WhatsApp, ERP, CRM, and carrier integrations, image verification, and processing exchanges or refunds.

## User stories and acceptance criteria

| ID | User story | Measurable criterion |
| --- | --- | --- |
| CA-01 | As a customer, I want to explain my claim in my own words. | Evaluate at least 32 labeled messages held out from agent tuning. Report counts by class and macro-F1 across the four categories. An abstention on a classifiable case counts as a false negative. Report referrals for ambiguous cases separately. The initial target is macro-F1 ≥ 0.80. |
| CA-02 | As a customer, I want to continue the conversation without repeating my order number. | After the user provides the order number in session A, a follow-up retrieves the same order. In session B, the same follow-up without context prompts for the order number. The visible history in B starts empty. |
| CA-03 | As a customer, I want to know where the agent gets its information. | A lookup for a known order returns its ID and the fields retrieved from the source, and the interface displays the raw JSON. An unknown ID prompts the user to correct it. The evidence panel and the n8n execution identify the query performed. |
| CA-04 | As a support team member, I want to reproduce the preliminary assessment. | Test a withdrawal request on the deadline, a late request, a public holiday, the distinction between a store-policy exchange and a defect claim, and incomplete evidence. The output includes the rule, version, reference date, assessment date, and outcome. Tests can fix the reference time. |
| CA-05 | As a customer, I want to review the case before filing it. | Submitting the form with the checkbox cleared sends zero write requests. Asking to file through chat or sending a direct POST without valid confirmation creates zero cases. A draft from another session also creates zero cases. |
| CA-06 | As a support team member, I want to recover the same case reference after a retry. | A valid confirmation creates one row. Repeating the confirmation returns the same `caseId`. Two concurrent confirmations of the same draft leave exactly one row. |
| CA-07 | As a customer, I want to view and download the result. | Streamlit displays its initial, loading, and error states in Spanish. After confirmation, the downloadable PDF includes the stored `caseId`, order, reason, filing date, and outcome. |
| CA-08 | As a team, we want to reproduce the demo on a clean installation. | Import the workflow, load synthetic data, configure credentials, and call the production webhooks from the Streamlit service. Record the versions, the contract test, and a rehearsal lasting ≤ 10 minutes. |

The proposed 60–80% screening rate is a business hypothesis. Report the sample size, method, and limitations alongside the evaluation results.

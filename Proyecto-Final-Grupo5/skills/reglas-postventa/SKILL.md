---
name: reglas-postventa
description: Model, explain, and test the preliminary decision rules for Jaleco Sports after-sales requests.
---

# After-sales rules

1. Infer the reason from the facts in the message. “No me quedó la talla” may be a store-policy exchange. “Pedí M y me enviaron S” calls for checking the product variant and points to a fulfillment issue.
2. Look up the order by ID and use its recorded delivery date. If the request type, order, or date is unclear or missing, ask for clarification or refer the case for review.
3. Look up the policy by category and effective date. Keep `policyId` and the version. For the right of withdrawal, consult Article 47 and model applicability, exceptions, and deadline calculations separately. For warranty claims, check Articles 7–11 of the Consumer Statute. Keep store-policy conditions as separate rules.
4. Calculate outcomes in code using local dates in `America/Bogota`. For a business-day rule, define the cutoff, calendar, and treatment of the reference date in both the policy and its tests.
5. Return the preliminary outcome, missing fields, dates, and rule. A missing photo or serial number may indicate pending evidence or the need for review. Filing creates a case record. Refund decisions remain a separate process.

Test the last eligible day, the first day after the deadline, a weekend, a public holiday, and a missing date or policy. Use synthetic data and compare the code's result with an independently established expected outcome.

Reference. [Law 1480 of 2011, Articles 7–11 and 47](https://normograma.dian.gov.co/dian/compilacion/docs/ley_1480_2011.htm)

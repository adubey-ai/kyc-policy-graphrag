# Document Retention and Audit Trail

Policy ID: POL-AUD-003. Owner: Internal Audit.

Every KYC packet and address-change case must retain: original document image, OCR or VLM extraction JSON, confidence scores, Maker identity, Checker identity, and CBS transaction id.

Retention is 8 years after account closure for OVD images and 10 years for EDD files.

The audit trail is append-only. Makers cannot delete an extraction after Checker view. Overrides must be recorded as exception EX-HITL-1 with a free-text reason.

IASW-style agentic pipelines that auto-extract fields are treated as Maker actions. They inherit Maker restrictions: no CBS write path.

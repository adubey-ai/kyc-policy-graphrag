# Account Address Change Policy

Policy ID: POL-CHG-014. Owner: Operations Control. Linked process: IASW maker-checker.

An address change on a savings or current account is a material KYC update. The customer must submit a fresh proof of address. Utility bill, Passport, and Aadhaar e-KYC are accepted. The bill date must be within 90 days.

The Maker agent (or branch Maker) extracts fields from the OVD using OCR or a vision-language model. Extracted fields with confidence below 0.82 cannot auto-fill the CBS address block.

The Checker must see the original document image, the extracted fields, and the previous address. The Checker is the only role with write access to the CBS customer-master address.

If the new address is in a high-risk pin code list HRC-22, the case is escalated to the Financial Crime Unit before Checker approval.

A PEP customer requesting an address change also triggers EDD even if onboarding EDD was completed earlier.

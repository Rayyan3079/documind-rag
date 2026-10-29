# Demo corpus

| File | Document | Pages | Source | License | In repo? |
|---|---|---|---|---|---|
| `NIST.CSWP.29.pdf` | NIST Cybersecurity Framework (CSF) 2.0, NIST CSWP 29, Feb 2024 | 32 | https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf | Public domain: a work of the U.S. federal government, not subject to copyright in the United States (attribution appreciated by NIST) | Yes |
| `NIST.SP.800-63B-4.pdf` | NIST SP 800-63B-4, Digital Identity Guidelines: Authentication and Authenticator Management, July 2025 | 129 | https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63B-4.pdf | Public domain (as above) | Yes |
| `pico-datasheet.pdf` | Raspberry Pi Pico Datasheet, release 21 (2026) | 31 | https://datasheets.raspberrypi.com/pico/pico-datasheet.pdf | CC BY-ND 4.0, © Raspberry Pi Ltd. The colophon also restricts use to "in conjunction with the Raspberry Pi products". | No: downloaded by `scripts/download_docs.py` |

Get the full corpus with:

```bash
python scripts/download_docs.py
```

Page numbers used in citations and in `eval/questions.jsonl` are **physical PDF page
indices** (1 = first page of the file), not the printed page labels. For example,
printed page 7 of the Pico datasheet is PDF page 8.

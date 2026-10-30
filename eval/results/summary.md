# DocuMind evaluation results

Run: 2026-10-30 00:15 UTC · corpus: 713 chunks from 3 PDFs · chunk size 800 / overlap 150 · embeddings `sentence-transformers/all-MiniLM-L6-v2`

## Retrieval (answerable questions, k=5)

| Mode | Hit rate@k | Hit rate@1 | MRR | Misses |
|---|---|---|---|---|
| semantic | 91% | 64% | 0.748 | pico-05, idm-06 |
| bm25 | 100% | 82% | 0.894 | - |
| hybrid | 100% | 86% | 0.898 | - |

## Answers (`claude-opus-5-5`, effort `medium`, hybrid retrieval, k=5; judge `claude-opus-5-5`)

| Metric | Value |
|---|---|
| Correct | 20/22 (91%) |
| Correct or partially correct | 21/22 (95%) |
| Wrong | 1 (of which false refusals: 1) |
| Answer cites an evidence page | 95% |
| Correct refusals on unanswerable questions | 4/4 (100%) |
| Latency p50 / p95 | 3.3 s / 6.1 s |
| Avg tokens per answer (in / out) | 1651 / 321 |
| Cost: answers / judge / total | $0.339 / $0.104 / $0.443 |

## Per-question verdicts

| ID | Type | Verdict | Cited evidence page | Latency |
|---|---|---|---|---|
| pico-01 | numeric | correct | yes | 4.6 s |
| pico-02 | numeric | correct | yes | 3.7 s |
| pico-03 | fact | correct | yes | 2.5 s |
| pico-04 | procedure | correct | yes | 3.4 s |
| pico-05 | numeric | wrong | no | 3.0 s |
| pico-06 | numeric | correct | yes | 2.0 s |
| pico-07 | table | correct | yes | 2.3 s |
| pico-08 | fact | partially_correct | yes | 4.6 s |
| pico-09 | fact | correct | yes | 2.3 s |
| pico-10 | table | correct | yes | 6.3 s |
| csf-01 | list | correct | yes | 4.2 s |
| csf-02 | list | correct | yes | 4.3 s |
| csf-03 | definition | correct | yes | 3.9 s |
| csf-04 | table | correct | yes | 1.8 s |
| csf-05 | fact | correct | yes | 2.0 s |
| csf-06 | list | correct | yes | 4.2 s |
| idm-01 | numeric | correct | yes | 2.6 s |
| idm-02 | fact | correct | yes | 2.4 s |
| idm-03 | numeric | correct | yes | 5.9 s |
| idm-04 | numeric | correct | yes | 3.6 s |
| idm-05 | fact | correct | yes | 2.8 s |
| idm-06 | reasoning | correct | yes | 6.1 s |
| neg-01 | unanswerable | correct_refusal | - | 0.9 s |
| neg-02 | unanswerable | correct_refusal | - | 1.7 s |
| neg-03 | unanswerable | correct_refusal | - | 1.4 s |
| neg-04 | unanswerable | correct_refusal | - | 3.3 s |

## Not fully correct

### pico-05: What is the recommended maximum load current on the Pico's 3V3 pin?
- Expected: Less than 300 mA.
- Got: I could not find this in the documents.
- Judge: wrong - false refusal (said not found)
- Retrieved: pico-datasheet.pdf p.12, pico-datasheet.pdf p.14, pico-datasheet.pdf p.8, pico-datasheet.pdf p.9, pico-datasheet.pdf p.13

### pico-08: What does GPIO23 control on the Pico, and what happens when it is driven high?
- Expected: It controls the RT6150 SMPS Power Save (PS) pin; driving it high forces PWM mode, which reduces output ripple at light load but lowers efficiency.
- Got: **What it controls:** GPIO23 is an output that controls the Power Save (PS) pin of the on-board RT6150 switch-mode regulator (SMPS) [pico-datasheet.pdf, page 21] [pico-datasheet.pdf, page 8].

**Default (low):** By default PS is low, so the regulator runs in Pulse Frequency Modulation mode [pico-datasheet.pdf, page 21]. At light loads, this mode saves considerable power by turning on the switching MOSFETs only occasionally to keep the output capacitor topped up [pico-datasheet.pdf, page 21].

**When driven high:** Setting PS high forces the regulator into Pulse Width Modulation (PWM) mode [pico-datasheet.pdf, page 21]. The datasheet describes this as power saving being disabled, and some of its current measurements were taken with GPIO23 pulled high [pico-datasheet.pdf, page 14] [pico-datasheet.pdf, page 15].

The documents do not say what practical benefit PWM mode offers or why you might choose to enable it.
- Judge: partially_correct - The system answer is right that GPIO23 controls the RT6150 PS pin and that driving it high forces PWM mode. It omits the trade-off in the reference: lower output ripple at light load but reduced efficiency. It even claims the documents do not give this benefit.
- Retrieved: pico-datasheet.pdf p.21, pico-datasheet.pdf p.8, pico-datasheet.pdf p.15, pico-datasheet.pdf p.14, pico-datasheet.pdf p.16


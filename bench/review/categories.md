# Free category labels in lean review replies

The sample contains 73 findings in 39 replies on 20 recorded reviews, with 44 distinct defects.
Join the ordered replay results to their adjudicated finding IDs by review, repeat and position.
Count every unordered pair within each adjudicated defect, including pairs across heads.
Compare the category strings exactly as written.
A fix counts when the adjudication records a corresponding code or documentation change; no recorded fix does not mean none exists.

| Category | Findings | Real at severity | Overstated | False | Recorded fix |
|---|---:|---:|---:|---:|---:|
| integrity | 15 | 12 | 1 | 2 | 12 |
| privacy | 11 | 9 | 2 | 0 | 8 |
| availability | 5 | 0 | 5 | 0 | 0 |
| audit | 5 | 4 | 1 | 0 | 4 |
| terminal | 4 | 1 | 3 | 0 | 2 |
| provenance | 3 | 2 | 1 | 0 | 3 |
| budget | 3 | 3 | 0 | 0 | 3 |
| accounting | 3 | 2 | 1 | 0 | 2 |
| idempotency | 3 | 0 | 3 | 0 | 0 |
| isolation | 2 | 0 | 1 | 1 | 0 |
| approval | 2 | 0 | 2 | 0 | 0 |
| bounds | 1 | 0 | 1 | 0 | 0 |
| coverage | 1 | 0 | 1 | 0 | 0 |
| permissions | 1 | 0 | 0 | 1 | 0 |
| testing | 1 | 1 | 0 | 0 | 0 |
| egress | 1 | 0 | 1 | 0 | 0 |
| disclosure | 1 | 1 | 0 | 0 | 1 |
| ordering | 1 | 1 | 0 | 0 | 1 |
| verification | 1 | 0 | 1 | 0 | 0 |
| visibility | 1 | 1 | 0 | 0 | 1 |
| documentation | 1 | 1 | 0 | 0 | 1 |
| policy | 1 | 1 | 0 | 0 | 1 |
| lifecycle | 1 | 1 | 0 | 0 | 1 |
| docs | 1 | 1 | 0 | 0 | 1 |
| consent | 1 | 0 | 1 | 0 | 0 |
| confirmation | 1 | 0 | 1 | 0 | 0 |
| authorization | 1 | 0 | 1 | 0 | 0 |
| liveness | 1 | 0 | 1 | 0 | 0 |

25 of 39 repeated pairs agreed on the label (64%).
There were 28 distinct labels.

Predict each finding from other defects only, by majority for severity and file, falling back to severity then the overall majority for ties or empty cells.
Add category within the severity and file cell, with the same fallback.

| Outcome | Correct without category | Correct with category | Rows with a category match |
|---|---:|---:|---:|
| Real mechanism (including overstated) versus false | 71/73 | 71/73 | 10/73 |
| Recorded fix versus none | 38/73 | 38/73 | 10/73 |

The sample does not establish a predictive signal beyond severity and file.
The removal rule needs both under 25% pair agreement and over 30 labels.
Keep the field.

### Pair agreement

| Defect | Pair | Categories | Agree |
|---|---|---|---|
| U01 | F01, F04 | privacy, privacy | Yes |
| U01 | F01, F18 | privacy, privacy | Yes |
| U01 | F01, F22 | privacy, privacy | Yes |
| U02 | F02, F06 | availability, availability | Yes |
| U02 | F02, F21 | availability, bounds | No |
| U01 | F04, F18 | privacy, privacy | Yes |
| U01 | F04, F22 | privacy, privacy | Yes |
| U02 | F06, F21 | availability, bounds | No |
| U13 | F12, F14 | integrity, integrity | Yes |
| U14 | F13, F15 | provenance, provenance | Yes |
| U06 | F16, F20 | availability, availability | Yes |
| U01 | F18, F22 | privacy, privacy | Yes |
| U25 | F28, F58 | egress, privacy | No |
| U25 | F28, F62 | egress, privacy | No |
| U26 | F29, F31 | terminal, terminal | Yes |
| U26 | F29, F64 | terminal, integrity | No |
| U27 | F30, F60 | privacy, privacy | Yes |
| U26 | F31, F64 | terminal, integrity | No |
| U28 | F32, F59 | disclosure, privacy | No |
| U28 | F32, F63 | disclosure, privacy | No |
| U30 | F33, F36 | accounting, budget | No |
| U31 | F34, F37 | accounting, budget | No |
| U32 | F35, F38 | ordering, integrity | No |
| U33 | F39, F41 | audit, audit | Yes |
| U33 | F39, F66 | audit, audit | Yes |
| U33 | F39, F70 | audit, audit | Yes |
| U33 | F41, F66 | audit, audit | Yes |
| U33 | F41, F70 | audit, audit | Yes |
| U39 | F44, F72 | approval, approval | Yes |
| U40 | F45, F47 | visibility, policy | No |
| U41 | F46, F49 | documentation, docs | No |
| U15 | F50, F52 | integrity, integrity | Yes |
| U21 | F53, F55 | consent, confirmation | No |
| U22 | F54, F57 | idempotency, idempotency | Yes |
| U25 | F58, F62 | privacy, privacy | Yes |
| U28 | F59, F63 | privacy, privacy | Yes |
| U37 | F65, F69 | integrity, integrity | Yes |
| U33 | F66, F70 | audit, audit | Yes |
| U38 | F67, F68 | terminal, terminal | Yes |

### First twenty findings in replay order

| Category | Title |
|---|---|
| privacy | Disclose that the helper receives inline operator answers |
| availability | Charge quote framing against the replay cap |
| integrity | Keep pending calls outside the idle endpoint summary |
| privacy | Disclose that helper fallback sends operator answers |
| provenance | Reject operator words on a harness-authored snapshot notice |
| availability | Charge per-message quote framing to the replay budget |
| integrity | Cover replies outside the cached idle-summary request |
| budget | Resolve the model against the full listing before looking up its price |
| accounting | Keep credit readings comparable across the fingerprint change |
| isolation | Discard model IDs from a different provider binding |
| privacy | Document that shared journals can be linked by account |
| integrity | A removal retry can delete a reviewer the operator did not confirm |
| provenance | Compare describes fresh files using the graph’s cached working-tree status |
| integrity | Retrying Remove can delete a different reviewer |
| provenance | Compare labels a fresh diff using cached worktree status |
| availability | Compaction bypasses the tool-result size bound |
| integrity | A capped answer can displace a newer operator correction |
| privacy | The summary-writer setting understates what reaches the helper |
| availability | A restart expands a capped pending result even with retention off |
| availability | The worker summary instruction has no budget for omitted answers |

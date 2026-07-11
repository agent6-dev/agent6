# stylebook - the SITEREP auditor

Implement one function in `audit.py`:

```python
def audit(text: str) -> list[str]
```

`text` is the full content of one SITEREP field report. `audit` returns the
code of every stylebook rule the report violates: a `list` of `str`, sorted
ascending, no duplicates (for example `["R01", "R07"]`). A fully compliant
report returns `[]`. The function is pure (no I/O) and uses only the Python
standard library.

## Report anatomy

A report is processed line by line (split on newlines). Each line is
classified by the first matching case, in this exact order of precedence:

1. **Blank** - the line is empty or contains only whitespace. Ignored by
   every rule.
2. **Comment** - the line's first character is `#`. Ignored by every rule.
   (A `#` after leading whitespace does not make a comment.)
3. **Continuation** - the line starts with two spaces. Governed by rule R09;
   its *text* is everything after the first two characters.
4. **Record** - anything else. Record lines carry the report's data as
   `TAG: value` pairs and are what most rules constrain.

## The rules

The ten rules live in `rules/rule01.md` through `rules/rule10.md`, one rule
per file, and each file is authoritative for its rule. Codes are `R01`
through `R10`, matching the file names. The files also pin down how rules
interact at the boundaries (which rule owns each corner case), so read ALL
ten files before implementing; spec.md deliberately restates none of them.

| code | concern                  |
|------|--------------------------|
| R01  | record line grammar      |
| R02  | required tags            |
| R03  | DATE validity            |
| R04  | CREW roster shape        |
| R05  | TEMP form and range      |
| R06  | DATE precedes samples    |
| R07  | sample ids               |
| R08  | note length              |
| R09  | continuation placement   |
| R10  | report terminator        |

A report can violate several rules at once. Report each violated rule's code
exactly once, no matter how many lines violate it, and never report a code
for a rule the report satisfies.

## Done when

`./verify.sh` passes (exit 0). Do not edit `test_audit.py` or `verify.sh`.
The starter tests are a small necessary-not-sufficient subset; the acceptance
battery checks every rule independently and in combination, so implement all
ten rules exactly as their files state them.

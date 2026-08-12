# Kept benchmark runs

Replay and demo material from real SWE-bench Verified runs, kept because
each shows something specific:

- `django-16950-spiral-escape/`: the recall-spiral trigger instance on a
  run that escaped it and produced a wrong patch anyway; full provider
  transcripts (the instance spirals in most runs, so an escape is the
  interesting trajectory).
- `sphinx-8035-grind/`: 46 tools and three red verifies ground down to a
  resolve at $1.03; run log and final patch (transcripts were 12M, not
  kept).

Full run corpora live outside the repo in the bench workspace; these are
the curated survivors.

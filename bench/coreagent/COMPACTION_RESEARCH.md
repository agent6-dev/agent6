# Context compaction for agent6: state of the art, mapped to the two-tier + DAG design

Research date: 2026-06-30. Target: agent6 0.0.15, `workflows/_compaction.py` + `workflows/loop.py` tier-1/tier-2, the curator-owned task DAG, and the unwired `memory.py` store.

## TL;DR

- agent6's tier-1 (elide oldest tool_result, keep the most recent few, leave a placeholder) is the **same primitive Anthropic ships** as `clear_tool_uses_20250919` (trigger 100k tokens, keep 3). This is validated by the strongest evidence in the field: observation masking "halves cost while matching, and sometimes slightly exceeding, the solve rate of LLM summarization" (The Complexity Trap, SWE-bench Verified, 5 model configs). **Keep tier-1; it is doing most of the work.**
- agent6's thresholds already adapt to the model window (45% / 80%, `models_cache.py`), so "make thresholds adaptive" is already done. The open lever is that effective context is far shorter than the claimed window (NoLiMa: GPT-4o holds quality only to ~8K of a 128K window), so **compacting earlier and at task boundaries helps quality, not just cost.**
- The biggest measured win in the literature is **structured memory that survives the restart**: Anthropic measured context-editing alone at +29% and context-editing + a file-based memory tool at +39% over baseline, with 84% fewer tokens on a 100-turn eval. agent6 already has the storage for this (`memory.py`, currently unwired) plus a durable DAG.
- The top 5 to prototype (all cheap, all A/B-able with completion rate / redundant re-reads / tokens-to-completion): (1) a deterministic facts ledger updated by tool results with no LLM call; (2) a keep-last-K-verbatim hybrid tier-2 instead of the hard 2-message restart; (3) compaction at verify-pass / DAG-subtask boundaries; (4) wire `memory.py` into compaction and the restart message; (5) lower / per-model-calibrate the compaction thresholds.

## How agent6's current design maps to the state of the art

What the evidence says agent6 is already doing right:

- **Tier-1 elision == Anthropic `clear_tool_uses`.** Verified defaults: trigger 100,000 input tokens, `keep=3` recent tool uses, `clear_tool_inputs=false` (args stay), each cleared result replaced with placeholder text, oldest-first. agent6 uses `keep_recent=2` and the same placeholder. This is also OpenHands' `ObservationMasking` / `RecentEvents` condensers and SWE-agent's `last_n_observations` (n=5). Convergent design across every serious agent.
- **Cheap-summary-then-restart at a high threshold == compaction** as Anthropic, Cline, Roo Code, Cursor, Codex all define it. agent6's 4-section summary (goal / tried+outcome / current state / next steps) matches OpenHands' condenser summary content (goals, progress, critical files, failing tests) and Cognition's "key details, events, and decisions."
- **Adaptive thresholds.** `models_cache.compaction_thresholds()` already sizes tier-1 at 45% and tier-2 at 80% of the model window (chars/token=4). Production triggers cluster in the same band: Anthropic API compaction default 150k tokens; context-editing 100k; Claude Code CLI ~83-95%; Roo Code default 100%; OpenCode at overflow minus output. agent6 is in range.
- **Durable task state across restart.** The DAG + checkoff is exactly the cross-system pattern: every serious agent externalizes a todo/notes object that survives summarization (Manus `todo.md`, Anthropic `NOTES.md` + memory tool, Claude Code `MEMORY.md`, Cline Focus Chain, Anthropic's long-running-harness JSON feature list edited only by flipping a per-feature `passes` flag). agent6's checkoff is a good version of this.

Where the evidence says agent6 has gaps:

1. **The hard 2-message restart is the most aggressive option in the field.** Everyone else keeps a recent verbatim tail: Anthropic "continue with the summary plus the five most recently accessed files"; Codex "recent user messages (~20k tokens) + summary"; LangChain ConversationSummaryBuffer keeps recent turns verbatim and rolls only the older prefix; Cursor keeps older messages as a searchable file. agent6 discards everything but `[task, summary]`. This is the direct cause of failure mode (b) re-reads and a likely contributor to (a) the empty post-restart turn (the model is dropped cold with no in-flight work to continue).
2. **The summariser sees a lossy view.** At tier-2 the transcript is rendered with each tool_result truncated to 800 chars and the whole thing tail-capped to 60k chars (`format_messages_tail_for_critic`). For a 200k-token model the tier-2 trigger is ~640k chars, so the summariser sees roughly the last 10% and never sees raw file contents. Early decisions are already gone before the summary is written. This argues for a rolling summary and/or a deterministic ledger that captures facts as they happen.
3. **A cheap separate summariser may hurt fidelity.** agent6 uses `summariser_provider` (a cheap model). Roo Code explicitly condenses with the *active* model because "using a different model... can degrade summary quality when the history includes tool calls." Worth an A/B.
4. **`memory.py` is built but unwired.** agent6 has `facts.md` / `decisions.md` / `preferences.md` (append-only, ULID-keyed, ripgrep-searchable) under the state dir, used only by `cli/memory_cmds.py`. It is not a tool, not in the prompt, not injected at restart. This is the single highest-evidence improvement sitting one wire away (+10 points in Anthropic's eval from adding a file memory on top of context editing).
5. **The DAG is underused as memory.** `TaskNode` already carries `relevant_paths`, `commit_sha`, `acceptance`, and `notes`. The free-text summary re-derives "files changed / latest commit sha" that the DAG could carry deterministically, so the summary could be smaller and more reliable.

## Evidence synthesis by question

### 1. WHAT to keep vs drop

- **Observation masking matches summarization at half the cost.** The Complexity Trap (arXiv 2508.21433, preprint, verified abstract): masking old observations "halves cost relative to the raw agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization" across 5 model configs on SWE-agent, replicated on OpenHands. A masking+summarization **hybrid** beats either alone by a further 7% / 11%. agent6's two-tier *is* that hybrid; the finding says lean on tier-1 and keep tier-2 light.
- **Keep first + last; the middle is where recall dies.** Lost in the Middle (TACL 2024): U-shaped accuracy, ~20 point drop for mid-context facts. StreamingLLM (ICLR 2024): 4 "attention sink" tokens at the front hold perplexity at 5.40 vs 5158 for a pure recency window. Practical rule: anchor the task/first message and keep a recent window; agent6 already keeps `messages[0]` and `keep_recent`.
- **Keep errors and decisions over raw file dumps.** Manus: "leave the wrong turns in the context... the model implicitly updates its beliefs." Anthropic compaction "preserves architectural decisions, unresolved bugs, and implementation details while discarding redundant tool outputs." OpenHands condenser summary keeps "critical files and failing tests." agent6 elides purely by recency, so it can drop the error that explains why a path was abandoned.
- **Salience / heavy-hitter eviction retains quality at a fraction of the tokens.** H2O (NeurIPS 2023): keeping ~20% of tokens (recent + heavy hitters) cuts KV memory 5x with no accuracy loss. SnapKV / PyramidKV: 88-92% KV reduction at near-full LongBench accuracy. These are KV-cache-level, not directly portable, but the principle (keep the few high-attention items, drop the bulk) transfers to choosing which tool_results to retain.

### 2. HOW to summarize

- **Structured / file memory beats prose, and adds on top of masking.** Anthropic measured: context editing alone +29%, context editing + file memory tool +39%, 84% fewer tokens over 100 turns (claude.com/blog/context-management, official). Mem0 (vendor arXiv): ADD/UPDATE/DELETE/NOOP fact reconciliation, ~26% over OpenAI memory on LoCoMo with ~90% fewer tokens. A-MEM (NeurIPS 2025): linked atomic notes roughly double multi-hop LoCoMo F1 vs MemGPT at ~87% fewer tokens. The pattern: maintain a small structured store of facts/decisions instead of re-summarizing the whole transcript each time.
- **Incremental / rolling beats one-shot.** LangChain ConversationSummaryBuffer rolls older turns into a running summary on a token trigger. A production A/B (EMNLP 2025 industry) found progressive note-taking cut handling time 3% on average, up to 9% on complex cases, vs one-shot end summarization. Cursor's RL-trained self-summarizer produces ~1,000-token summaries (vs >5,000 baseline) and cut long-horizon error 50% at one-fifth the tokens. Relevant because agent6's tier-2 one-shot only sees the last 10% of history.
- **Virtual memory paging (MemGPT/Letta).** Tiered context (RAM = window, recall + archival = external), self-edited under "memory pressure" (warn at 70%, flush ~50% at 100%); lifted deep multi-session recall from 32.1% to 92.5% on a synthetic benchmark. The actionable nugget for agent6 is the pressure ladder (warn early, evict in stages) rather than a single hard restart.
- **Retrieval scoring recipe.** Generative Agents (UIST 2023): score = recency + importance + relevance, all weights 1, recency decay 0.995/step, importance a 1-10 LLM rating. This is the canonical salience formula if agent6 ever ranks what to keep rather than using pure recency.

### 3. WHEN to compact

- **Performance degrades well below the hard limit, so compacting earlier helps quality.** NoLiMa (ICML 2025): effective length (>=85% of base score) is mostly 1K-8K despite 128K+ windows; GPT-4o falls 99.3% -> 69.7% at 32K. "Context Length Alone Hurts LLM Performance Despite Perfect Retrieval" (arXiv 2510.05381): 13.9-85% degradation from length alone even when all needed tokens are retrievable and distractors are masked; reciting the evidence first (turning long context into short) recovered up to ~4% on RULER. Chroma context rot: all 18 frontier models degrade with length, and a single distractor measurably hurts. This is the case for lowering agent6's 80% trigger, especially for the small/open models agent6 targets.
- **Compact at task boundaries, not mid-task.** Context-Folding (arXiv 2510.11967, verified abstract): branch into a subtask, fold it to a concise outcome summary on completion; 10x smaller active context and "significantly outperforms models that rely on summarization-based context management." Active Context Compression (arXiv 2601.07190): agent-declared focus boundaries make a "sawtooth" context, -22.7% tokens at flat accuracy (small n). LangChain guidance: compress at structural boundaries and right after token-heavy tool calls return. agent6 has perfect natural boundaries the others lack: a verify-command pass and a DAG node going `passed`.
- **Don't compact too often.** Repeated passes cause "summarization drift" (low-frequency but critical details vanish after ~3 cycles, illustrative not measured). Favors anchored, append-only, recall-first compaction over frequent full rewrites.

### 4. What real coding agents do (documented)

- **Anthropic API:** `compact_20260112` (server-side, default trigger 150k tokens, drops all prior blocks after the compaction block); `clear_tool_uses_20250919` (trigger 100k, keep 3, placeholder); `memory_20250818` (client-side file tool, `/memories`, persists across conversations, auto-prompt "VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE... ASSUME INTERRUPTION").
- **Claude Code:** `CLAUDE.md` loaded every session and re-read from disk after `/compact`; auto memory `MEMORY.md` (first 200 lines / 25KB) loaded each session; CLI auto-compact ~83-95% (env `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`); "micro-compaction" clears old tool results via `cache_edits` to preserve the cache prefix.
- **Manus:** KV-cache hit rate is the top metric (cached 0.30 vs uncached 3.00 USD/MTok, 10x; ~100:1 input:output); append-only prefix-stable context; mask tool logits instead of editing tool defs; filesystem as unlimited external memory with restorable compression (drop page content, keep the URL); rewrite `todo.md` each step to recite the goal into recent attention; keep failed actions in context.
- **OpenHands:** condenser zoo: `LLMSummarizingCondenser` (keep_first, max_size; summarize the middle), `ObservationMasking`, `RecentEvents`, `AmortizedForgetting`, `LLMAttention`. `keep_first` is an explicit attention sink.
- **SWE-agent:** `last_n_observations` history processor (n=5), "Old environment output: (n lines omitted)"; breaks prompt caching.
- **Aider:** tree-sitter repo map ranked by PageRank over the import graph, capped at `--map-tokens` (default 1k), expanded when no files are in chat.
- **Cline:** auto-compact reuses the prompt cache (costs ~one tool call); `/smol`; Focus Chain todo list persists through summarization.
- **Roo Code:** Intelligent Context Condensing (on by default, percent trigger, custom prompt); condenses with the *active* model because a different model degrades quality when history has tool calls.
- **Cursor:** summarizes older messages and exposes prior history as a *searchable file reference* so the agent can recover details dropped from the summary.
- **Cognition (Devin):** single-threaded context, a dedicated (sometimes fine-tuned, smaller) LLM that compresses history into key details/events/decisions; argues against multi-agent for coding because parallel agents act on conflicting assumptions.

## (a) Ranked candidate improvements for agent6

Difficulty is relative to the existing two-tier + DAG code. "Novel" marks ideas tailored to agent6 beyond the literature.

| # | Technique | Evidence / source | Expected benefit | Difficulty in agent6 | How to measure |
|---|-----------|-------------------|------------------|----------------------|----------------|
| 1 | **Deterministic facts ledger** updated in the tool-dispatch path (no LLM call): record symbol/grep hits as `path:line`, every verify outcome (cmd + pass/fail), files touched, latest commit sha; render it verbatim into the restart message and keep it out of elision. **Novel.** | Anthropic "lightweight identifiers loaded at runtime"; just-in-time context; structured note-taking; Mem0 fact store. agent6 already has symbol tools + verify. | Kills failure (b) re-reads: the model has file:line and prior verify results without re-grepping/re-reading. Zero added latency or cost, can't hallucinate. | Medium. New `_FactsLedger` updated in `tools/dispatch.py`; rendered in `_summarise_and_restart` and surfaced after restart. No model call. | Redundant re-reads (identical-arg read_file/grep/symbol calls), tokens-to-completion, completion rate. |
| 2 | **Keep-last-K-turns verbatim hybrid tier-2**: summarize only the older prefix, append the summary, then keep the last K balanced message-pairs verbatim instead of restarting to `[task, summary]`. **Novel adaptation.** | LangChain ConversationSummaryBuffer; Anthropic "summary + 5 most recently accessed files"; Codex "recent ~20k tokens + summary"; Cursor; Complexity Trap hybrid beats either alone. | Fixes (b) re-reads and likely (a) empty turn (recent reasoning + any in-flight tool result remain to continue from). Recent chain-of-thought is no longer lost. | Medium. The loop already compacts only at a balanced boundary, so slicing the last K *pairs* keeps tool_use/tool_result pairing intact. Change `messages[:] = [original, restart] + tail`. | Empty-turn-post-restart rate (telemetry already emitted), re-reads, completion, tokens. |
| 3 | **Compact at verify-pass / DAG-subtask boundaries**, not only at the 80% char threshold: when a verify passes or a DAG node goes `passed` and context is over a lower watermark, fold that subtask to its outcome. **Novel adaptation of Context-Folding to agent6's existing structure (no RL needed).** | Context-Folding (10x smaller active context, beats summarization); Active Context Compression sawtooth (-22.7% tokens, flat accuracy); LangChain compress-at-boundaries. | Targets (c) wandering: forces a conclusion-checkpoint at clean points; the summary is crisp because a subtask just finished. Smaller active context = better recall (NoLiMa). | Medium. Add a boundary trigger in the loop's verify / `update_status(passed)` path that calls a fold (reuse tier-2 with the keep-last-K tail). | Completion on long/vague tasks, "wander" (turns since last DAG progress), tokens-to-completion. |
| 4 | **Wire `memory.py` into compaction + restart**: auto-append durable facts/decisions at tier-2, inject a memory digest into the restart message and system prompt, and add a "view memory first" instruction. Optionally expose as a tool. | Anthropic context-editing +29% vs +memory +39%, 84% token cut (official, measured); Anthropic memory-tool auto-prompt; Claude Code MEMORY.md survives /compact; Mem0; A-MEM. | Largest measured delta in the field. Carries decisions/dead-ends across *multiple* restarts (the summary only carries one). Storage already exists. | Medium. `memory.py` storage is built and unwired; add write at tier-2 + read into prompt/restart. Decide auto-populate vs tool. | Completion, dead-end repeats (re-trying a reverted approach), re-reads, multi-restart runs especially. |
| 5 | **Lower / per-model-calibrate the compaction thresholds**: drop the 80% tier-2 fraction (and 45% tier-1) and/or size from an *effective* window table, not the claimed window. | NoLiMa (effective length 1-8K vs 128K); Context Length Alone Hurts (13-85%); Chroma context rot; effective-context-falls-short (~half training length). | Better recall and fewer mid-context misses, especially for the small/open models agent6 targets. Cheapest possible A/B. | Low. Change `_SUMMARISE_FRACTION` / `_DROP_FRACTION` in `models_cache.py`, or add a per-model effective-window multiplier. | Sweep the fraction; completion + tokens-to-completion + re-reads per setting. Pure one-knob A/B. |
| 6 | **Salience-keep in tier-1**: never elide a tool_result that a later edit references (same path) or that contains an error / failed verify; elide oldest *low-salience* first. **Novel.** | H2O heavy-hitters (keep 20%, no loss); Anthropic "5 most recently accessed files"; Manus keep-failed-actions; Generative Agents importance score. | Fewer re-reads of the file currently being edited; preserves the error that explains an abandoned path. | Low-Medium. Add a salience predicate to `compact_old_tool_results` (skip results whose path appears in a later `edit_file` tool_use, or whose body looks like an error). | Re-reads, dead-end repeats, completion. |
| 7 | **Lean on the DAG to shrink the summary**: store per-task `notes` (files touched, outcome) via the curator, render the DAG (with `relevant_paths` + `commit_sha`) deterministically into the restart, and cut the summariser prompt to "narrative delta only." **Novel.** | Anthropic long-running harness (JSON feature list, edit only the `passes` flag); Cline Focus Chain; Manus todo recitation. agent6 `TaskNode` already has the fields. | Smaller, more reliable summary (the DAG carries structured state); less for a weak summariser to get wrong; reinforces "DAG is authoritative." | Low-Medium. Curator `notes` writes + a deterministic DAG renderer in the restart; trim `CONTEXT_SUMMARY_SYSTEM_PROMPT`. | Summary token size, completion, re-reads, checkoff accuracy. |
| 8 | **Restorable elision placeholder**: embed the originating tool args (path/offset, command) and a pointer to the ledger entry in the tier-1 placeholder, instead of "re-call with the same args." | Manus restorable compression (drop content, keep URL); Anthropic lightweight identifiers; Cursor searchable history. | When the model does re-read, it targets the right args on the first try; fewer wasted calls. Cheap. | Low. Enrich `ELISION_PLACEHOLDER` to include the tool name + args of the elided call. | First-try re-read success, re-reads, tokens. |
| 9 | **Structured JSON summary schema** instead of free-text prose: `{goal, attempts:[{what,outcome}], files_changed, best_score, head_sha, dead_ends, next_steps}`. | Cursor RL structured summary; Cognition key details/events/decisions; Mem0 structured ops. | More parseable (merge with ledger), and a structured handoff may reduce the (a) empty-turn confusion. | Low. Prompt change + light parse; agent6 already parses a fenced checkoff block. | Empty-turn rate, completion, summary fidelity (spot-check). |
| 10 | **A/B the summariser model**: summarize with the worker model, not the cheap `summariser_provider`. | Roo Code: condense with the active model; a different model "can degrade summary quality when the history includes tool calls." | May cut the (a) empty-turn and improve fidelity; quantifies the cheap-summariser tradeoff. | Low. Already a config seam (`summariser_provider`). | Empty-turn rate, completion, summary fidelity, summary cost. |
| 11 | **Pressure ladder instead of one hard restart**: warn / shed in stages (e.g. tighten tier-1 keep_recent, then fold a subtask, then full restart) as context climbs. | MemGPT memory pressure (warn 70%, flush 50% at 100%); OpenCode protective buffer; staged condensers. | Smoother degradation; avoids the cliff where everything is dropped at once. | Medium. Add an intermediate stage between tier-1 and tier-2 in `_maybe_compact`. | Completion, tokens, re-reads, frequency of full restarts. |
| 12 | **Rolling incremental summary** updated every N iterations so early history is captured before the 60k tail-cap drops it. | LangChain rolling buffer; incremental-beats-one-shot (3-9%); Cursor self-summary. | Tier-2 no longer loses the first 90% of history; better long-run fidelity. | Medium-High. Extra periodic model calls (cost), or a deterministic merge. | Summary fidelity on long runs, completion, added summary cost. |
| 13 | **Per-model effective-window table** feeding the thresholds (refinement of #5). | NoLiMa per-model effective lengths; RULER. | Right-sizes compaction per model family; protects weak models. | Medium. Add an effective-window map / multiplier in `models_cache.py`. | Completion + re-reads per model. |
| 14 | **KV-cache-aware elision**: batch tier-1 edits to a cache boundary / prefer append-only, since editing old messages invalidates the cache from that point. | Manus 10x cached-vs-uncached, 100:1 I/O; Cline cache-reuse summary; Claude Code micro-compaction `cache_edits`. | Lower $ per run; tier-1 currently rewrites old content and breaks the cache prefix each time. | Medium-High. Provider-cache-dependent; restructure when/where elision mutates the list. | Cached-token ratio, USD per run. |
| 15 | **Attention-sink keep-first in tier-1**: also pin the first tool_result (initial repo/AGENTS.md read) as a sink, not only the last 2. | StreamingLLM (4 sink tokens, 5.40 vs 5158 PPL); OpenHands `keep_first`; Lost in the Middle. | Marginal; agent6 already pins `messages[0]` + DAG, so the front is mostly anchored. | Low. `keep_first` arg in `compact_old_tool_results`. | Completion, re-reads (expect small effect). |

### Ranking rationale

Top of the table maximizes (objective-metric A/B-ability) x (evidence) / (difficulty) and each targets a distinct known failure: #1 ledger -> (b) re-reads; #2 keep-last-K -> (a) empty turn + recent-reasoning loss; #3 boundary compaction -> (c) wandering; #4 memory.py -> dead-end repeats + cross-restart knowledge (the single biggest measured delta); #5 thresholds -> tokens + quality via the cheapest one-knob sweep. #6-#10 are low-cost follow-ons. #11-#15 are higher-effort or lower-marginal-value.

## (b) Top 5 to prototype first, and why

1. **Deterministic facts ledger (#1).** Cheapest high-impact change and the cleanest A/B. It runs in the tool-dispatch path with no model call, so it adds zero latency, zero cost, and cannot hallucinate. It attacks failure (b) head-on: the model re-reads because the *content* was elided, but a `path:line` + last-verify-result line is tiny and never needs eliding. Metric: redundant re-reads (count tool calls whose args equal an earlier call), expected to drop sharply; secondary, tokens-to-completion.

2. **Keep-last-K-verbatim hybrid tier-2 (#2).** The hard 2-message restart is the field's most aggressive choice and the likely root of both (a) and (b). Every other agent keeps a recent verbatim tail; the Complexity Trap shows a masking+summary hybrid beats either alone. The loop already compacts at a balanced boundary, so keeping the last K pairs is safe and small. Metric: empty-turn-post-restart rate (already in telemetry) and re-reads.

3. **Compact at verify-pass / DAG-subtask boundaries (#3).** agent6 has a structural advantage no general chat agent has: a verify command and a DAG with `passed` transitions. Context-Folding shows folding at subtask boundaries gives a 10x smaller active context and beats rolling summarization, and you get the boundary structure deterministically without the paper's RL. It directly attacks (c): forcing a checkpoint when a subtask finishes is the opposite of wandering. Metric: completion on long/vague tasks and "turns since last DAG progress."

4. **Wire `memory.py` into compaction + restart (#4).** Highest measured evidence anywhere in this report: Anthropic's own numbers put file memory at +10 points over context editing alone (+39% vs +29%) with 84% fewer tokens. agent6 already has the store; it is unwired. It is the only candidate that carries decisions and dead-ends across *multiple* restarts, where a single one-shot summary cannot. Metric: dead-end repeats (re-trying a reverted approach) and completion on multi-restart runs.

5. **Lower / per-model-calibrate thresholds (#5).** The cheapest experiment of all: one constant. The degradation evidence (NoLiMa, Context Length Alone Hurts, context rot) says agent6's 80% trigger lets quality rot before compaction fires, especially on the small/open models it targets. Sweep the fraction and read completion + tokens. Do this one first as a baseline calibration, then layer #1-#4 on top. Metric: completion and tokens-to-completion across threshold settings.

These five compose: #1 and #4 make early/aggressive compaction (#2, #3, #5) safe, because the facts and decisions survive deterministically even when prose is dropped.

## Measurement harness (objective A/B)

agent6 already emits `loop.compact.*` telemetry and has bench harnesses (`bench/agents`, `bench/swebench`) and verify + DAG status. Instrument three metrics, hold the model and task set fixed, flip one candidate at a time:

- **Task completion rate.** DAG root `passed` and/or final verify pass at run end. Primary quality metric.
- **Redundant re-reads.** In `tools/dispatch.py`, hash each tool call's `(name, args)`; count calls whose hash matched an earlier call in the same run. Break out pre- vs post-compaction. This is the direct signal for failure (b) and the cleanest discriminator for #1, #2, #6, #8.
- **Tokens-to-completion.** Sum input+output tokens (agent6 tracks budget/usage) to first DAG-root-pass. Cost/efficiency metric; pairs with completion to catch quality-for-tokens regressions.

Secondary: empty-turn-post-restart rate (for #2, #9, #10), summary token size (for #7, #12), dead-end repeat count (for #4, #6), cached-token ratio (for #14). Run each on a fixed suite of long tasks that actually trigger tier-2 (the short bench tasks will not exercise compaction).

## Risks and caveats

- **Compacting earlier is lossy unless paired with structured memory.** Do not ship #5 alone; land #1/#4 first so facts survive. "Summarization drift" (rare critical details vanishing after a few passes) is the failure to watch.
- **Evidence quality varies.** Strongest (peer-reviewed): StreamingLLM, H2O, Lost in the Middle, NoLiMa, Generative Agents, Reflexion, MemGPT, LongMemEval, Agentless, AutoCodeRover. Preprints (provisional): Complexity Trap, Context-Folding, Active Context Compression, A-MEM, Context Length Alone Hurts. Vendor/official-doc (authoritative for their own systems, not independent benchmarks): Anthropic, Manus, Cursor, Cline, Roo Code, OpenHands, Cognition, Mem0, Chroma. The Anthropic +29/+39/84% numbers are an internal eval; directionally strong, not externally replicated.
- **KV-cache interaction (#14).** agent6's tier-1 edits old messages in place, which can invalidate provider prompt caches from the edit point. Worth measuring before adding more in-place rewrites; append-only (Manus) is the cheaper shape if the provider caches.
- **Keep-last-K pairing.** Only slice at a balanced boundary (every tool_use has its tool_result), which the loop already guarantees at compaction time. Slicing mid-pair orphans a tool call and some providers reject it.

## (c) Sources

Peer-reviewed / conference:
- https://arxiv.org/html/2309.17453v3 (StreamingLLM, ICLR 2024)
- https://arxiv.org/abs/2306.14048 (H2O, NeurIPS 2023)
- https://arxiv.org/abs/2305.17118 (Scissorhands, NeurIPS 2023)
- https://aclanthology.org/2024.tacl-1.9/ and https://arxiv.org/abs/2307.03172 (Lost in the Middle, TACL 2024)
- https://arxiv.org/html/2502.05167v1 (NoLiMa, ICML 2025)
- https://openreview.net/forum?id=kIoBbc76Sy (RULER)
- https://arxiv.org/abs/2410.18745 (Why Effective Context Length Falls Short)
- https://ar5iv.labs.arxiv.org/html/2304.03442 (Generative Agents, UIST 2023)
- https://arxiv.org/abs/2303.11366 (Reflexion, NeurIPS 2023)
- https://ar5iv.labs.arxiv.org/html/2308.10144 (ExpeL, AAAI 2024)
- https://arxiv.org/abs/2409.07429 (Agent Workflow Memory, COLM 2025)
- https://arxiv.org/abs/2305.10250 (MemoryBank, AAAI 2024)
- https://ar5iv.labs.arxiv.org/html/2310.08560 (MemGPT, ICML 2024)
- https://arxiv.org/abs/2310.05029 (MemWalker)
- https://arxiv.org/abs/2402.09727 (ReadAgent, ICML 2024)
- https://arxiv.org/abs/2109.10862 (Recursively Summarizing Books)
- https://arxiv.org/abs/2410.10813 (LongMemEval, ICLR 2025)
- https://arxiv.org/html/2502.12110v1 (A-MEM, NeurIPS 2025)
- https://arxiv.org/abs/2407.01489 (Agentless, FSE 2025)
- https://arxiv.org/abs/2404.05427 (AutoCodeRover, ASE 2024)
- https://aclanthology.org/2025.emnlp-industry.140/ (incremental vs one-shot summarization, EMNLP 2025)
- https://arxiv.org/abs/2310.06201 (Selective Context, EMNLP 2023)

Preprints (provisional):
- https://arxiv.org/abs/2508.21433 (The Complexity Trap: observation masking vs summarization)
- https://arxiv.org/abs/2510.11967 (Context-Folding / FoldGRPO)
- https://arxiv.org/abs/2601.07190 (Active Context Compression / Focus)
- https://arxiv.org/abs/2510.05381 (Context Length Alone Hurts Despite Perfect Retrieval)
- https://arxiv.org/abs/2505.02709 (Goal drift correlates with context length)
- https://arxiv.org/html/2404.14469v1 (SnapKV)
- https://arxiv.org/html/2406.02069v1 (PyramidKV)
- https://arxiv.org/abs/2406.06110 (Recurrent Context Compression)
- https://arxiv.org/abs/2507.02259 (MemAgent)
- https://arxiv.org/html/2504.19413v1 and https://mem0.ai/research-3 (Mem0)
- https://arxiv.org/abs/2506.08098 (Cognitive Weave)
- https://arxiv.org/abs/2412.14161 (TheAgentCompany)

Vendor engineering / official docs:
- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- https://claude.com/blog/context-management (context editing +29/+39%, 84% token cut)
- https://platform.claude.com/docs/en/build-with-claude/context-editing (clear_tool_uses: trigger 100k, keep 3)
- https://platform.claude.com/docs/en/build-with-claude/compaction (compact_20260112, default 150k)
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool (memory_20250818)
- https://code.claude.com/docs/en/memory (CLAUDE.md / MEMORY.md, survives /compact)
- https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents (JSON feature list, passes flag)
- https://www.anthropic.com/engineering/multi-agent-research-system (sub-agent distillation, 15x tokens)
- https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus (KV-cache, todo.md, keep-errors, filesystem memory)
- https://cognition.com/blog/dont-build-multi-agents (single-thread, compression LLM)
- https://cursor.com/blog/self-summarization (RL self-summarizer, ~1k-token summaries)
- https://docs.cursor.com/en/agent/chat/summarization (history as searchable file)
- https://docs.cline.bot/features/auto-compact and https://cline.bot/blog/how-to-think-about-context-engineering-in-cline (Focus Chain)
- https://roocodeinc.github.io/Roo-Code/features/intelligent-context-condensing (condense with active model)
- https://github.com/OpenHands/OpenHands/blob/main/config.template.toml and https://docs.openhands.dev/sdk/guides/context-condenser (condenser zoo)
- https://www.openhands.dev/blog/openhands-context-condensensation-for-more-efficient-ai-agents (54% vs 53%, <half cost)
- https://swe-agent.com/latest/reference/history_processor_config/ (last_n_observations)
- https://aider.chat/docs/repomap.html and https://aider.chat/2023/10/22/repomap.html (PageRank repo map)
- https://www.trychroma.com/research/context-rot (18-model degradation)
- https://rlancemartin.github.io/2025/06/23/context_engineering/ and https://www.langchain.com/blog/context-engineering-for-agents (write/select/compress/isolate)
- https://reference.langchain.com/python/langchain-classic/memory/summary_buffer/ConversationSummaryBufferMemory and https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware
- https://gist.github.com/badlogic/cd2ef65b0697c4dbe2d13fbecb0a0a5f (Claude Code / Codex / OpenCode / Amp trigger reverse-engineering)
- https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/ (LLMLingua)
- https://www.augmentcode.com/guides/ai-agent-loop-token-cost-context-constraints

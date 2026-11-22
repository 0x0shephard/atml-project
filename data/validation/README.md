# Answer-detector validation (2026-11-22)

This folder checks whether the rule in `src/crs/extract/reasoning_scan.py` finds real cases of the model stating the correct answer inside its reasoning.

- `make_sample.py` draws the sample (seed 0) from the scored R1-1.5B runs: 10 failed chains per setting where the rule fired, plus 5 failed chains where it did not fire but the gold value appears somewhere in the text. That gives 117 chains and 306 spans.
- `labeller_A.md` and `labeller_B.md` are what the two independent labellers saw. They show the problem, the reference answer, and each span with context. The labellers did not see the setting, the outcome or the rule's verdict, and the two files share 16 chains (42 spans).
- `labels_A.json` and `labels_B.json` hold the labels. The question for each span was: *does the solver state the reference answer as its answer to the whole problem here, even tentatively?*
- `agreement.py` scores the rules against the labels.
- `human_check.csv` has 40 spans (20 YES, 20 NO by the labellers) for people on the team to label blind. `human_check_key.json` maps each row back to its span.

Both labellers were Claude models, not people. Treat their labels as a strong-LLM reference, in the same role as Lotfi et al.'s GPT-5 judge, until the human check is done.

Results: the labellers agreed on 42/42 shared spans.

| Rule | Precision (chain level) | Recall |
|---|---|---|
| Loose (`had_correct_scan`) | 72% | misses about 30% of failed chains where the gold appears only as a bare mention |
| Strict (`had_correct_strict`: boxed / "the answer is") | 100% | 29% |
| Loose, dropping conclusion values copied from the problem (now the default) | about 80% | about 98% of the loose rule's real hits |

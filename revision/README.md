# Supplementary major-revision analyses

These analyses were specified after the original primary outcomes were known. They add exactly ten classifier fits: five collection-source LightGBM diagnostics and five source-trained Logistic Regression fits using SGD. The experimental seeds are the five consecutive integers from 2026 through 2030. All training used CPU. No hyperparameter search, bootstrap, new VT retrieval or original-primary retraining occurred.

## Compact verification

From the candidate root, with the packages in `revision/requirements.txt`:

```powershell
python code/reproduce_review.py --root . --output ../verify_primary_new --profile numerical
python revision/code/reproduce_revision.py --output ../verify_revision_new
python -O revision/code/test_revision_checks.py
```

Both output directories must be new. The first command checks the unchanged primary/historical artifact. The second verifies included revision identities, linear validation thresholds, all new per-seed metrics and mean/SD summaries, structural weights, group separation and small-fixture model inference. It uses the 1,024 fixed, label-blind original feature rows. It does not refit models or evaluate new scientific conditions. The source-diagnostic fixture is the intersection with each corresponding held-out test partition. Linear scores are finite logits, not probabilities; see the recorded float32 fixture tolerance policy.

## Results and interpretation

- `source_diagnostic_*`: collection source is the prediction target. Within-VT-label metrics condition test records; fitting was pooled. These are not malware-detection rates or a causal confounding estimate.
- `common_support_*`: at least 20 records from each source in label/file-group/DLL/size cells. Only label 0 has support. Rates use recorded, inspectable weights and do not estimate deployment prevalence.
- `label_sensitivity_*`, `count_band_*`, `label_by_age_*`: primary scores/thresholds, positive cutoff 5/10/15 and disjoint count bands. No changes to negative labels; FPR invariance is mathematical, not label validation.
- `exact_relatedness_*`: exclude every source neighbor at TLSH distance <=30 (including length) from the primary cohort. This is a sensitivity only, not a change to the primary result. The stored witness is not necessarily nearest.
- `linear_*`: the fixed source-only linear baseline, source test, primary and key subgroups. Paired differences are percentage points; no p-values or equivalence claim.

In weighted common-support rows, `FP`, `TP` and class denominators remain ordinary record counts, while `FPR` and `TPR` use the supplied record weights. A weighted rate must not be reconstructed by dividing those unweighted counts. Undefined positive-class metrics remain null when no positive records qualify.

## Full-input reproduction recipes

The compact artifact does not include the full feature matrices or source TLSH index. `config/revision_analysis.json` identifies all original matrix, metadata and split inputs by SHA-256. Original executed scripts and training claims/receipts are retained. A full-input replay is optional and was not rerun merely for packaging.

To repeat five fits for one role on separately acquired identical inputs, create a JSON map from every input ID in `revision_analysis.json` to its local path, then use a new output directory:

```powershell
python revision/code/reproduce_training.py --role linear --input-map input_paths.json --output ../linear_full_replay_new
python revision/code/reproduce_training.py --role source --input-map input_paths.json --output ../source_full_replay_new
```

These commands train models and are deliberately separate from compact verification. Each checks input hashes before fitting and preserves the archived model configuration. They are recipes, not claims that a second full training run was executed.

`exact_relatedness.py` is the executed full-source search. Its complete input identities, search specification, source-index pointer, target witnesses and verification receipts are retained. The compact checker reconstructs performance from the included near-neighbor ledger; it does not certify absence of neighbors anew without the large source index. No approximate retrieval is used in the recorded full search.

## Technical history

An unsigned-integer subtraction error in an initial synthetic TLSH test was rejected before real execution. Explicit integer casts resolved it; the final implementation passed original-library distance and exhaustive small-reference tests before producing the real ledger. That rejected log is retained in the revision technical history. This did not alter an existing scientific result or trigger a second detector fit.

The ten classifier claims are one execution each. Some linear seeds were dispatched in parallel, within local CPU and memory limits; scheduling does not change the frozen configuration. A duplicate/incomplete claim is refused rather than silently retrained. Training receipts retain actual epochs and any optimization-budget warnings.

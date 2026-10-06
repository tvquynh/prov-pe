# PROV-PE: Windows PE features and provenance

## Resource and intended use

The resource contains 779,619 SHA-linked Windows PE records separately collected and curated from a malware-candidate collection and a benign-reference collection. It complements EMBER2024 for cross-dataset research. Separate collection does not imply disjoint sample identities, independent malware families or representative deployment prevalence. EMBER2024 provides the reference representation, official overlap inventory and source domain for the demonstrated evaluation.

## Inventory and evaluation views

The inventory retains 196,191 operational label-0 records, 553,322 label-1 records and 30,106 unresolved records. The label rule uses the stored malicious detection count: zero for label 0, at least five for label 1, and unresolved otherwise or when the report is unavailable. These are operational VT labels, not independently adjudicated ground truth.

The primary evaluation contains 742,273 records: 189,114 negatives and 553,159 positives. Eligibility requires a binary operational label, a named Win32, Win64 or .NET group, an available positive first-submission timestamp and no exact overlap with official EMBER2024 PE membership. All excluded and unresolved records remain in the inventory. The 6,517 inventory overlaps and 6,437 sequential primary-cohort exclusions refer to different scopes; they must not be interchanged.

## Representation

Stored feature records vectorize to the 2,568-dimensional EMBER v3 layout. The qualified primary view is the first 2,480 coordinates, excluding the 88-coordinate PE-format-warning group. Both domains use this same projection before fitting or evaluation. The full layout and qualified view have different roles. The warning diagnostic establishes a specific order-dependent mismatch; removing that group does not prove universal equivalence across extractors, versions or platforms. Seven categorical coordinates retain their original positions.

## Provenance, time and relatedness

Metadata link records to collection source, report availability, operational-label information, report timestamps, eligibility masks and overlap controls. Collection-source diagnostics distinguish MalwareBazaar from the benign-reference collection within this corpus. They do not distinguish EMBER2024 from this corpus. A report retrieved in September 2026 can reflect an older analysis. VT first submission is not verified first occurrence or collection time. The resource is not a synchronized longitudinal panel.

SHA exclusions, identical-TLSH controls and the declared TLSH-distance sensitivity have distinct scopes. The distance search covers the stated official source index at distance at most 30. Neither digest equality nor distance exclusion establishes family independence. Common structural support is observed only for the negative class under the implemented cell rule; resampling cannot supply missing positive-class combinations.

## Demonstrated and conditional reuse

The supplied evaluation demonstrates source-only reference training, source-selected thresholds, external evaluation and fixed-score cohort sensitivities. All reported seed results are retained. New learners need access to the full permitted features and their own training-overlap, preprocessing and threshold controls. Matched, temporal or distribution-shift designs require an explicitly justified cohort and adequate metadata/support; they are not automatically validated by the existing demonstration.

## Access and reproducibility

The release includes feature JSONL, Parquet, metadata and the separate research artifact with code, saved outputs and verification recipes. Full source training features remain obtainable from the authoritative EMBER2024 release. See README.md for archive contents, licenses and limits. No scientific computation was rerun for packaging.

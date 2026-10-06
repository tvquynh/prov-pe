# PROV-PE: Windows PE features and provenance

Research resource for *PROV-PE: A provenance-qualified Windows PE dataset with EMBER v3 features for cross-dataset malware evaluation*.

Authors: Van-Quynh Trinh; De-Thu Huynh; Trong-Thua Huynh (corresponding);
Minh Thang Nguyen. Contact: thuaht@ptit.edu.vn.

## Current release status

Code is public in this repository. PROV-PE version 2.0.0 was published on
6 October 2026 with all 11 release files (6.27 GB):
[download the dataset and research artifacts on Zenodo](https://doi.org/10.5281/zenodo.23157392).
Public file access and release checksums have been verified; see RELEASE_STATUS.json.

## Data and scope

The inventory contains 779,619 records: 196,191 operational label-0,
553,322 label-1 and 30,106 unresolved records. The primary evaluation view
contains 742,273 records. Stored features have 2,568 coordinates; the
qualified warning-free evaluation uses the first 2,480 coordinates in both
domains. Labels use stored VT malicious counts (0 / at least 5 / unresolved),
not independent adjudication. Retrieval time is not analysis time. Collection
source is associated with class composition; the corpus is not a deployment
prevalence estimate. See DATA_CARD.md for scientific limitations.

## Release layout

- `PROV-PE_JSONL_v2.0.0.zip`: all 256 JSONL feature shards, aligned provenance
  sidecars and original build/validation manifests (779,619 inventory rows).
- `PROV-PE_Parquet_Metadata_v2.0.0.zip`: full feature Parquet, sample provenance,
  primary target manifest, historical relatedness ledger and configuration.
- `PROV-PE_Qualification_Inputs_v2.0.0.zip`: frozen metadata inputs for eligibility/overlap
  replay, including attributed EMBER2024 reference membership and TLSH.
- `PROV-PE_Research_Artifact_v2.0.0.zip`: saved models, scores, result tables, fixtures,
  executed source code, configurations, environment and historical receipts.
- `ZENODO_SHA256SUMS.txt`: checksums of Zenodo release files. Each archive also contains a
  member manifest. Extract each archive into its own named directory.

Code is mirrored at https://github.com/tvquynh/prov-pe .
The software citation is recorded in CITATION.cff; the data release uses v2.0.0. The earlier Zenodo v1 contains
only an identifier placeholder and is not the research-file release.

## Use

Read the feature Parquet with pandas/pyarrow. JSONL readers should glob only
`shard_*.jsonl` at the archive root, not provenance sidecars recursively.
Join using SHA-256, preserve recorded row order when using saved scores,
and distinguish the full inventory from the primary-cohort manifest.
Original PE binaries and raw VT reports are not part of this release.

For saved-score verification, extract `PROV-PE_Research_Artifact_v2.0.0.zip`, install
`revision/requirements.txt`, and run from the extracted directory:

```powershell
./reproduce_major_revision.ps1 -Python python -OutputRoot ../fresh_verification
```

Use a fresh output directory. This checks archived outputs and small model
fixtures; it does not train. Full-training recipes require separately obtained
identical EMBER2024 source features and the frozen source split assets named
in the configurations. Original absolute execution paths in provenance are
historical identifiers, not portable defaults. No training, bootstrap,
new VT request or scientific-result computation was run for this release.

## Optional report retrieval

For future retrieval of existing VirusTotal reports, use
[`data_preparation/vt_single_key.py`](data_preparation/vt_single_key.py).
It accepts one API key, sends requests sequentially, preserves quota state
across restarts and stops on HTTP 429. See the
[client instructions](data_preparation/VT_CLIENT_README.md) for limits and
offline tests. This optional client was not used to acquire the published
dataset. Reproduction uses the frozen metadata and saved outputs; it does
not require new API requests. Historical verification receipts describe
their original snapshots and may identify files not distributed here.

## Licensing

Authors' code: MIT. Authors' feature data and metadata: CC BY 4.0.
Third-party components retain their licenses; see THIRD_PARTY_NOTICES.md.

## Published file naming

Release assets use the `PROV-PE_` prefix. See `FILENAME_MAP.json` for previous names and public Parquet member names. Numbered JSONL shards, code paths and historical receipts retain their original names. Renaming changes no scientific data, labels or results. The complete release is published on Zenodo under DOI 10.5281/zenodo.23157392.

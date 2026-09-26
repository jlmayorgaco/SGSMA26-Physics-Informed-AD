# Citation audit

Final status: **PASS**

An independent fresh-context reviewer checked all citations in `main.tex`
against `references.bib` and external publisher, DOI, standards, institutional,
DBLP, arXiv, OSTI, or JMLR records. The audit covered three axes: source
existence, bibliographic metadata, and support for the sentence in which the
source is cited.

- Cited keys: 18
- Bibliography entries: 18
- Missing keys: 0
- Uncited entries: 0
- Sources that could not be verified: 0
- Unresolved context or metadata defects after revision: 0

The full audit returned WARN. Four corrections were made in that pass:

1. `yuan2021` is now cited for nonstationarity and missing or corrupt
   measurements, not for the broader claim of domain mismatch.
2. The sentence citing `ganjkhani2024` now distinguishes graph,
   knowledge-driven, and multi-source methods.
3. `sgsmaPolicy2026` no longer assigns authorship to an unsupported organizing
   committee; the corporate author is recorded as `SGSMA 2026`.
4. `pedregosa2011` now uses the canonical title capitalization, author
   diacritics, and JMLR article number 85.

A fresh-context delta audit on 11 August 2026 then checked the revised citation
contexts and bibliography. It requested three further corrections before
returning PASS:

1. `pandey2020` is described as engineered statistical and physics-based
   feature work, avoiding a broader characterization than the source supports.
2. `sgsmaGuide2026` records the 18 February 2026 revision date and its official
   organizer URL.
3. `sgsmaPolicy2026` records the 1 June 2026 revision date and its official
   organizer URL.

| Key | Existence | Metadata | Context | Final verdict |
|---|---:|---:|---:|---|
| `phadke2008` | PASS | PASS | PASS | KEEP |
| `ieee2018` | PASS | PASS | PASS | KEEP |
| `sgsmaGuide2026` | PASS | PASS | PASS | KEEP (corrected) |
| `sgsmaPolicy2026` | PASS | PASS | PASS | KEEP (corrected) |
| `pandey2020` | PASS | PASS | PASS | KEEP (context corrected) |
| `yuan2021` | PASS | PASS | PASS | KEEP (context corrected) |
| `taghipour2023` | PASS | PASS | PASS | KEEP |
| `yuan2023gnn` | PASS | PASS | PASS | KEEP |
| `liu2023` | PASS | PASS | PASS | KEEP |
| `li2019` | PASS | PASS | PASS | KEEP |
| `yildiz2024` | PASS | PASS | PASS | KEEP |
| `ahmed2024` | PASS | PASS | PASS | KEEP |
| `ganjkhani2024` | PASS | PASS | PASS | KEEP (context corrected) |
| `li2024knowledge` | PASS | PASS | PASS | KEEP |
| `athay1979` | PASS | PASS | PASS | KEEP |
| `cui2021andes` | PASS | PASS | PASS | KEEP |
| `geurts2006` | PASS | PASS | PASS | KEEP |
| `pedregosa2011` | PASS | PASS | PASS | KEEP (corrected) |

Primary verification records included Springer/DOI for `phadke2008` and
`geurts2006`, IEEE Standards Association for `ieee2018`, the official SGSMA
competition materials for the guide and policy, DOI/publisher records for the
journal papers, and the official JMLR record for `pedregosa2011`. Fifteen DOI
links resolved and matched the cited metadata; the JMLR article legitimately has
no DOI. The local organizer documents were checked against their source PDFs.

Audit provenance and hashes for the synchronized source, bibliography, citation
contexts, and final PDF are recorded in `CITATION_AUDIT.json`. The full-audit
trace remains in `.aris/traces/citation-audit/2026-08-07_run01/`; the 11 August
delta re-audit was performed by a separate fresh-context reviewer. The final
slide-narrative revision changed no citation-bearing sentence; source and PDF
hashes were refreshed after the synchronized compile.

The 12 August external-review revision narrowed the citation-bearing language
(`non-anticipative`, supervised `mixed`, and target-domain calibration) without
adding a source or expanding any attributed claim. Citation contexts and final
artifact hashes were refreshed after compilation.

The subsequent title and voice revision changed presentation rather than claim
scope. Citation-bearing sentences were updated in the context record, the full
key set remained unchanged, and source/PDF hashes were refreshed after the
six-page build.

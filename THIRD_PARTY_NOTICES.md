# Third-party notices

This repository contains original project code and adaptations of published research implementations. The root MIT license applies to Shayan Bemanian's original contributions. Upstream-derived material remains subject to the applicable upstream license.

## RLMRec

- Project: [HKUDS/RLMRec](https://github.com/HKUDS/RLMRec)
- Inspected revision: `22413752246de3dee8ab0d509f7f7a8889080f95`
- License: Apache License 2.0, reproduced in `LICENSES/RLMRec-Apache-2.0.txt`
- Relevant adapted files include `models/rlmrec_faithful.py` and related LightGCN/RLMRec compatibility code.
- Modifications include local model registration, explicit objective ablations, device portability, corrected evaluation-cache invalidation, reproducibility metadata, and the experiment/analysis framework.

No raw RLMRec datasets, generated profiles, semantic embeddings, or checkpoints are redistributed. Obtain them from the upstream release under its applicable terms.

## FREEDOM

The static semantic graph framework is inspired by the frozen-graph and graph-sampling ideas described in:

> Zhou, X. and Shen, Z. A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation. ACM Multimedia, 2023.

The public export excludes the local literal transcription of FREEDOM graph operators, the upstream-parity model variant, and the associated parity test. The included `SIG` implementation deliberately differs in topology, normalization, fusion, pruning rate, modality, and objective; those differences are documented in the hybrid-framework manuscript.

## Springer LNCS class

`papers/llncs.cls` is distributed with its original notices. `papers/LLNCS-README.md` records the template's licensing information.

## Manuscripts

The included manuscripts are unpublished drafts by Shayan Bemanian. The software license does not grant separate republication rights to the manuscript text beyond the permissions provided by law.

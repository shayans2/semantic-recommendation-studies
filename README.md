# Semantic Recommendation Studies

Code and two manuscripts by Shayan Bemanian on semantic representations for recommendation.

## Papers

- [A Hybrid Framework for Recommendation via LLM Semantic Anchoring and Static Graph Filtering](papers/hybrid-llm-semantic-anchoring-static-graph-filtering.pdf)
- [Revisiting RLMRec: User-Profile Alignment, Early Stopping, and Reproducibility](papers/revisiting-rlmrec-reproducibility.pdf)

The hybrid framework improves Recall@20 over full RLMRec on Amazon-book and Steam and performs worse on Yelp. The second study examines RLMRec's user-profile alignment loss and the effect of premature early stopping.

## Code

The main hybrid implementation is `models/sig.py`. The experiment entry point is `experiments/compare_sig_vs_lightgcn.py`.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Raw datasets and semantic embeddings are not included. Obtain the Amazon-book, Yelp, and Steam inputs from [RLMRec](https://github.com/HKUDS/RLMRec) and place them under `data/`.

Run the test suite with:

```bash
python -m pytest -q
```

Editable LaTeX sources and finished PDFs are in `papers/`.

RLMRec-derived files retain their Apache-2.0 attribution. See `THIRD_PARTY_NOTICES.md`.

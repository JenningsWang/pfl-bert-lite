# Public Repo Scope

This repository is the public, simplified version of a larger research prototype.

## Included In Public

- graph-guided topology updates driven by data scale, label skew, structural similarity, and gradient alignment,
- decentralized aggregation via a consensus matrix,
- constraint-aware subnetwork selection for heterogeneous clients,
- LoRA-based BERT model assembly,
- a minimal local fine-tuning loop,
- a deployment-oriented FastAPI route,
- tests, config files, and helper.

## Intentionally Omitted

- full supernet training code,
- fairseq-based translation code,
- packet-level communication simulation branches,
- private datasets and checkpoints,
- long experiment history and internal plotting notebooks,`r`n- the original D2D federated learning scripts; only refactored topology, aggregation, and orchestration ideas remain.

## Why This Split Exists

The public version is designed to be readable, reviewable, and interview-friendly. It keeps the architecture and the most defensible technical modules without exposing every research detail or every historical dependency.


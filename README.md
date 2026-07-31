# PFL-BERT Lite

`PFL-BERT Lite` is a public-facing research prototype for personalized federated fine-tuning of BERT-like models under non-IID data, compute heterogeneity, and decentralized communication constraints.

This repository distills three ideas into a compact, public-view-ready codebase:

1. Constraint-aware model selection inspired by supernet / NAS style compression.
2. BERT-style fine-tuning with LoRA adapters for parameter efficiency.
3. Dynamic graph-guided decentralized aggregation with a consensus matrix that depends on data scale, label skew, structural similarity, and gradient alignment.

The goal is not to reproduce every research detail from the original internal experiments. The goal is to expose the architecture, core algorithms, and engineering surface you can discuss in a GitHub repo.

## What Is Included

- A clean `src/` package layout.
- Dynamic topology construction in `src/pfl_bert_lite/graph/topology.py`.
- Consensus aggregation in `src/pfl_bert_lite/aggregation/consensus.py`.
- LoRA-BERT model assembly in `src/pfl_bert_lite/modeling/lora_bert.py`.
- Constraint-aware subnetwork selection in `src/pfl_bert_lite/nas/constraint_search.py`.
- Local fine-tuning loop in `src/pfl_bert_lite/training/local_update.py`.
- A federated orchestration layer in `src/pfl_bert_lite/training/federated_trainer.py`.
- A deployment-oriented FastAPI route in `src/pfl_bert_lite/serving/app.py`.


## Repository Layout

```text
pfl-bert-lite/
├── configs/
├── docs/
├── notebooks/
├── scripts/
├── src/pfl_bert_lite/
│   ├── aggregation/
│   ├── data/
│   ├── graph/
│   ├── modeling/
│   ├── nas/
│   ├── serving/
│   └── training/
└── tests/
```

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the lightweight dry-run demo:

```bash
python scripts/train_demo.py --config configs/demo_sst2.json
```

Run the pure-Python test suite:

```bash
python -m unittest discover -s tests
```

Start the inference stub:

```bash
uvicorn pfl_bert_lite.serving.app:app --app-dir src --host 0.0.0.0 --port 8000
```

## Core Paths For Applications

- Core model code: `src/pfl_bert_lite/modeling/lora_bert.py`
- Core local training code: `src/pfl_bert_lite/training/local_update.py`
- Federated aggregation logic: `src/pfl_bert_lite/training/federated_trainer.py`
- Topology update logic: `src/pfl_bert_lite/graph/topology.py`
- Consensus matrix logic: `src/pfl_bert_lite/aggregation/consensus.py`
- Deployment route: `src/pfl_bert_lite/serving/app.py`

## Scope Notes

This repo intentionally omits:

- the full original fairseq / YOCO training stacks,
- large packet-loss simulation branches,
- full dataset download pipelines,
- private experiment logs and checkpoints,`r`n- the original decentralized FL source files are withheld; only framework-level abstractions and refactored ideas are exposed.



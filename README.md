# Derm1M: A Million‑Scale Vision‑Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology

[![ArXiv](https://img.shields.io/badge/arXiv-2503.14911-b31b1b)](https://arxiv.org/abs/2503.14911)
[![License](https://img.shields.io/badge/License-CC%20BY–NC%204.0-green)](#license)
[![Downloads](https://img.shields.io/badge/Data-Coming%20Soon-blue)](#data-access)

<div align="center">
  <img src="overview.png" alt="Overview of Derm1M Dataset" width="80%"/>
</div>

## ✨ TL;DR

**Derm1M** brings **1,029,761 dermatologist‑curated image–text pairs**—257× more than any previous dermatology vision‑language corpus—covering **390 skin conditions** and **130 clinical concepts** organised in a four‑level expert ontology. The dataset’s rich contextual captions (mean = 41 tokens) include Fitzpatrick skin‑tone metadata and other clinical details, enabling explainable multimodal learning, zero‑/few‑shot diagnosis, cross‑modal retrieval and visual question answering in realistic settings.

## 📦 What’s inside?

| Aspect                                                      | Derm1M                                                                                                  |
| :---------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| Total pairs                                                 | **1,029,761**                                                                                           |
| Skin conditions                                             | **390** (4 hierarchy levels)                                                                            |
| Clinical concepts                                           | **130**                                                                                                 |
| Avg. caption length                                         | **41 words**                                                                                            |
| Image sources                                               | YouTube (51 k vids), PubMed (566 k figs), Medical forums (49 k posts), Public datasets, Teaching slides |
| Fitzpatrick tone metadata \| Present (I–VI, expert‑audited) |                                                                                                         |
| Ontology                                                    | Structured JSON + graph (disease ⇌ concept ⇌ finding)                                                   |

## 🔑 Key Features

* **Scale that matters** – 1 M+ multimodal examples enable training CLIP‑style encoders from scratch or adapting large vision‑language models.
* **Comprehensive coverage** – 390 conditions span inflammatory, infectious, neoplastic and genodermatoses, mirroring real‑world prevalence.
* **Rich context & concepts** – Descriptions embed history, symptoms, anatomic sites, Fitzpatrick tone and 130 physician‑understandable concepts for trustworthy reasoning.
* **Expert‑curated ontology** – Four board‑certified dermatologists built a hierarchical knowledge graph to supervise multi‑granular learning.
* **Benchmark‑ready splits** – Standard train/val/test splits plus 8 downstream datasets for plug‑and‑play evaluation.

## 🏗️ Repository Layout

```text
dataset_root/
├── images/                # JPEG & PNG images
├── captions.jsonl         # text + meta per image
├── ontology.json          # disease & concept hierarchy
├── splits/                # train/val/test indices
└── docs/
    ├── dataset_card.md
    └── LICENSE
```

## 🚀 Pre‑trained Models: **DermLIP**

We provide two CLIP‑style checkpoints trained from scratch on **Derm1M**:

| Model ID            | Vision Encoder | Text Encoder | Zero‑shot Avg† | R\@10 I→T (hold‑out) |
| :------------------ | :------------- | :----------- | :------------: | :------------------: |
| **DermLIP‑B/16**    | ViT‑B/16       | GPT77        |     56.1 %     |        40.7 %        |
| **DermLIP‑PanDerm** | PanDerm‑B      | PMB256       |   **58.8 %**   |      **59.9 %**      |

†Average accuracy across HAM10000, Fitzpatrick17K, PAD‑UFES‑20 and Daffodil datasets (Table 2) with retrieval metrics from Table 4. fileciteturn1file15turn1file6

Weights, inference scripts and data loaders will be released upon paper acceptance.

## 📊 Key Benchmarks

* **Zero‑shot disease classification (4 datasets):** 58.8 % accuracy, outperforming BiomedCLIP by **≈ 14.7 pp** fileciteturn3file2.
* **Few‑shot (1 % labels) linear probing:** 58.6 % accuracy vs. MONET 53.0 % (**+5.6 pp**) fileciteturn2file2.
* **Cross‑modal retrieval – Derm1M hold‑out:** Recall\@10 (I→T) 59.9 %, **3.6 ×** BiomedCLIP fileciteturn3file14.
* **Cross‑modal retrieval – SkinCAP external:** Recall\@10 (I→T) 20.2 %, **+6.0 pp** over MONET fileciteturn3file14.

*All numbers correspond to the DermLIP PanDerm‑B/PMB256 configuration reported in Tables 2–4 of the paper.*

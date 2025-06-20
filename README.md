# Derm1M: A Million-Scale Vision–Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology
[![arXiv](https://img.shields.io/badge/arXiv-2503.14911-b31b1b)](https://arxiv.org/abs/2503.14911)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-green)](#-license)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-blue)](#-citation)

> ⚠️ **Repository under construction**  
> Weights, training & inference scripts, and dataset access will be released upon paper acceptance.

---

## ✨ TL;DR
**Derm1M** brings **1,029,761 dermatological image–text pairs**—∼257 × larger than any previous dermatology V + L corpus—covering **390 skin conditions** and **130 clinical concepts** arranged in a four-level expert ontology. The rich captions (mean ≈ 41 tokens) embed metadata and physician-friendly concepts, enabling trustworthy multimodal learning, zero-/few-shot diagnosis, cross-modal retrieval and VQA in realistic settings.

<p align="center">
  <img src="overview.png" alt="Derm1M overview" width="800">
</p>

| **Aspect** | Derm1M |
|-----------|--------|
| **Total image–text pairs** | 1,029,761 |
| **Unique images** | 403,563 |
| **Refined medical captions** | 403,563 |
| **Ontology-based captions** | 403,563 |
| **Concept-based captions** | 222,635 |
| **Skin conditions** | 390 (4-level ontology) |
| **Clinical concepts** | 130 |
| **Average caption length** | ≈ 41 words |
| **Image sources** | YouTube (∼51 k videos), PubMed (∼566 k figures), medical forums (∼49 k posts), public dermatology datasets, teaching-slide collections |
| **Ontology structure** | Structured JSON + graph (disease ⇌ concept ⇌ finding) |

---

## 🔑 Key Features
- **Scale that matters** – 1 M+ pairs enable CLIP-style dermatology VL models.  
- **Comprehensive coverage** – 390 conditions spanning inflammatory, infectious, neoplastic & genodermatoses.  
- **Rich context & concepts** – 130 clinician-understandable concepts enhance explainability.  
- **Expert-curated ontology** – Four board-certified dermatologists built a four-level hierarchy.  
- **Benchmark-ready splits** – Eight downstream datasets for plug-and-play evaluation.  

---

## 🏗️ Repository Layout
```text
dataset_root/
├── images/            # JPEG / PNG images
├── captions.csv       # text + metadata per image
├── ontology.json      # disease & concept hierarchy
└── splits/            # train/val indices

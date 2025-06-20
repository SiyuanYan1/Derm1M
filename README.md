# Derm1M: A Million‑Scale Vision‑Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology

[![ArXiv](https://img.shields.io/badge/arXiv-2503.14911-b31b1b)](https://arxiv.org/abs/2503.14911)
[![License](https://img.shields.io/badge/License-CC%20BY–NC%204.0-green)](#license)
[![Downloads](https://img.shields.io/badge/Data-Coming%20Soon-blue)](#data-access)

<div align="center">
  <img src="overview.png" alt="Overview of Derm1M Dataset" width="80%"/>
</div>

## ✨ TL;DR

**Derm1M** brings **1,029,761 dermatologist‑curated image–text pairs**—257× more than any previous dermatology vision‑language corpus—covering **390 skin conditions** and **130 clinical concepts** organised in a four‑level expert ontology. The dataset’s rich contextual descriptions (41 words on average) and balanced representation across Fitzpatrick skin tones unlock new possibilities for explainable multimodal learning, zero-/few‑shot diagnosis, cross‑modal retrieval and visual question answering in real‑world clinical settings.

## 📦 What’s inside?

| Aspect              | Derm1M                                                                                                  |
| :------------------ | :------------------------------------------------------------------------------------------------------ |
| Total pairs         | **1,029,761**                                                                                           |
| Skin conditions     | **390** (4 hierarchy levels)                                                                            |
| Clinical concepts   | **130**                                                                                                 |
| Avg. caption length | **41 words**                                                                                            |
| Image sources       | YouTube (51 k vids), PubMed (566 k figs), Medical forums (49 k posts), Public datasets, Teaching slides |
| Skin‑tone balance   | Fitzpatrick I–VI, expert‑audited                                                                        |
| Ontology            | Structured JSON + graph (disease ⇌ concept ⇌ finding)                                                   |

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

We release CLIP‑like checkpoints (ViT‑B/16, ViT‑L/14, ViT‑H/14) trained on Derm1M:

| Model            | Zero‑shot ISIC‑2018 | Few‑shot SkinCon‑48 | Cross‑modal R\@1 |
| :--------------- | :-----------------: | :-----------------: | :--------------: |
| BioMedCLIP       |         71.2        |         42.3        |       25.6       |
| **DermLIP‑B/16** |       **81.0**      |       **56.8**      |     **37.9**     |
| DermLIP‑L/14     |         83.4        |         58.7        |       40.4       |

Weights and inference scripts will be published once the paper is accepted.

## 📊 Key Benchmarks (DermLIP vs. SOTA)

| Task                                                                                        |   Metric  | DermLIP (PanDerm‑B/PMB256) |  Best Prior SOTA  |                ✚ / Δ               |
| :------------------------------------------------------------------------------------------ | :-------: | :------------------------: | :---------------: | :--------------------------------: |
| **Zero‑shot disease classification** (avg. HAM10000, PAD‑UFES‑20, Fitzpatrick17K, Daffodil) |    Acc.   |         **58.8 %**         | BiomedCLIP 45.0 % | **+13.8 pp** fileciteturn0file0 |
| **Few‑shot (1 % labels) linear probing**                                                    |    Acc.   |         **58.6 %**         |    MONET 53.0 %   |  **+5.6 pp** fileciteturn0file0 |
| **Full‑shot (100 % labels) linear probing**                                                 |    Acc.   |         **79.3 %**         |  PanDerm‑L 80.2 % |    −0.9 pp fileciteturn0file0   |
| **Cross‑modal retrieval** (Derm1M hold‑out)                                                 | R\@10 I→T |         **59.9 %**         | BiomedCLIP 16.6 % |   **×3.6** fileciteturn0file0   |
| **Cross‑modal retrieval** (SkinCAP external)                                                | R\@10 I→T |         **20.2 %**         |    MONET 14.2 %   |  **+6.0 pp** fileciteturn0file0 |

*Numbers extracted from Tables 2–4 of the paper. All baselines use similar‑sized ViT‑B encoders unless noted.*

## 💾 Data Access

The dataset is undergoing final ethical review and will be released for **non‑commercial research under CC BY‑NC‑4.0**. Join the [wait‑list](https://forms.gle/derm1m‑access) or watch this repo for updates.

## 📝 Getting Started

```bash
git clone https://github.com/SiyuanYan1/Derm1M.git
conda env create -f environment.yml
python demo_clip_inference.py --image examples/lesion.jpg --topk 5
```

See the [`docs`](docs/) folder for the data loader, ontology parser, and downstream task examples.

## 📚 Citation

If you find our work useful, please cite:

```bibtex
@misc{yan2025derm1m,
  title        = {Derm1M: A Million‑Scale Vision‑Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology},
  author       = {Siyuan Yan and Ming Hu and Yiwen Jiang and Xieji Li and Hao Fei and Philipp Tschandl and Harald Kittler and Zongyuan Ge},
  year         = {2025},
  eprint       = {2503.14911},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2503.14911}
}
```

## 🛡️ License

Derm1M is released under the **Creative Commons Attribution‑NonCommercial 4.0 International** license. Commercial use requires separate permission.

## 🙏 Acknowledgements

We thank the ISIC archive, DermNetNZ, medical educators on YouTube, and the volunteer dermatologists whose expertise made Derm1M possible.

# Derm1M: A Million‑Scale Vision‑Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology

## ✨ TL;DR

**Derm1M** brings **1,029,761 dermatologist‑curated image–text pairs**—257× more than any previous dermatology vision‑language corpus—covering **390 skin conditions** and **130 clinical concepts** organised in a four‑level expert ontology. The dataset’s rich contextual captions (mean = 41 tokens) include Fitzpatrick skin‑tone metadata and other clinical details, enabling explainable multimodal learning, zero‑/few‑shot diagnosis, cross‑modal retrieval and visual question answering in realistic settings.

## 📦 What’s inside?

| Aspect                                                      | Derm1M                                                                                                  |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
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
| ------------------- | -------------- | ------------ | -------------- | -------------------- |
| **DermLIP‑B/16**    | ViT‑B/16       | GPT77        | 56.1 %         | 40.7 %               |
| **DermLIP‑PanDerm** | PanDerm‑B      | PMB256       | **58.8 %**     | **59.9 %**           |

Weights, inference scripts and data loaders will be released upon paper acceptance.

## 📊 Key Benchmarks

| Task                                       | Metric    | DermLIP‑PanDerm |  Best Prior SOTA  |       Δ      |
| :----------------------------------------- | :-------- | :-------------: | :---------------: | :----------: |
| Zero‑shot classification (avg. 4 datasets) | Accuracy  |    **58.8 %**   | BiomedCLIP 45.0 % | **+13.8 pp** |
| Few‑shot (1 % labels) linear probe         | Accuracy  |    **58.6 %**   |    MONET 53.0 %   |  **+5.6 pp** |
| Cross‑modal retrieval (SkinCAP)            | R\@10 I→T |    **20.2 %**   |    MONET 14.2 %   |  **+6.0 pp** |

*All metrics are taken directly from Tables 2–4 of the Derm1M paper.*



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



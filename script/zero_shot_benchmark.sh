# DermLIP - ViT-B-16
python src/main.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size=1024 \
    --zeroshot-eval1=data/meta/PAD-ZS.csv \
    --zeroshot-eval2=data/meta/HAM-ZS.csv \
    --zeroshot-eval3=data/meta/F17K-ZS.csv \
    --zeroshot-eval4=data/meta/Daffodil-ZS.csv \
    --csv-label-key label \
    --csv-img-key image_path \
    --csv-caption-key 'truncated_caption' \
    --model 'hf-hub:redlessone/DermLIP_ViT-B-16'

# DermLIP - PanDerm-base-w-PubMed-256
python src/main.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size=1024 \
    --zeroshot-eval1=data/meta/PAD-ZS.csv \
    --zeroshot-eval2=data/meta/HAM-ZS.csv \
    --zeroshot-eval3=data/meta/F17K-ZS.csv \
    --zeroshot-eval4=data/meta/Daffodil-ZS.csv \
    --csv-label-key label \
    --csv-img-key image_path \
    --csv-caption-key 'truncated_caption' \
    --model 'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256'

# MAKE
python src/main.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size=1024 \
    --zeroshot-eval1=data/meta/PAD-ZS.csv \
    --zeroshot-eval2=data/meta/HAM-ZS.csv \
    --zeroshot-eval3=data/meta/F17K-ZS.csv \
    --zeroshot-eval4=data/meta/Daffodil-ZS.csv \
    --csv-label-key label \
    --csv-img-key image_path \
    --csv-caption-key 'truncated_caption' \
    --model 'hf-hub:xieji-x/MAKE'

# -----
# ViT-L-14 - OPENAI
python src/main.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size=1024 \
    --zeroshot-eval1=data/meta/PAD-ZS.csv \
    --zeroshot-eval2=data/meta/HAM-ZS.csv \
    --zeroshot-eval3=data/meta/F17K-ZS.csv \
    --zeroshot-eval4=data/meta/Daffodil-ZS.csv \
    --csv-label-key label \
    --csv-img-key image_path \
    --csv-caption-key 'truncated_caption' \
    --model ViT-L-14 \
    --pretrained openai

# PMC-OA
python src/main.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size=1024 \
    --zeroshot-eval1=data/meta/PAD-ZS.csv \
    --zeroshot-eval2=data/meta/HAM-ZS.csv \
    --zeroshot-eval3=data/meta/F17K-ZS.csv \
    --zeroshot-eval4=data/meta/Daffodil-ZS.csv \
    --csv-label-key label \
    --csv-img-key image_path \
    --csv-caption-key 'truncated_caption' \
    --model 'hf-hub:ryanyip7777/pmc_vit_l_14'

# BiomedCLIP
python src/main.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size=1024 \
    --zeroshot-eval1=data/meta/PAD-ZS.csv \
    --zeroshot-eval2=data/meta/HAM-ZS.csv \
    --zeroshot-eval3=data/meta/F17K-ZS.csv \
    --zeroshot-eval4=data/meta/Daffodil-ZS.csv \
    --csv-label-key label \
    --csv-img-key image_path \
    --csv-caption-key 'truncated_caption' \
    --model 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
# DermLIP - ViT-B-16
python src/main.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size=1024 \
    --zeroshot-eval1=meta-files/zs/pad-zero-shot-test.csv \
    --zeroshot-eval2=meta-files/zs/HAM-official-7-zero-shot-test.csv \
    --zeroshot-eval3=meta-files/zs/f17k-113-zero-shot-test.csv \
    --zeroshot-eval4=meta-files/zs/daffodil-5-zero-shot-test.csv \
    --csv-label-key label \
    --csv-img-key image_path \
    --csv-caption-key 'truncated_caption' \
    --model 'hf-hub:redlessone/DermLIP_ViT-B-16'

# DermLIP - PanDerm-base-w-PubMed-256
python src/main.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size=1024 \
    --zeroshot-eval1=meta-files/zs/pad-zero-shot-test.csv \
    --zeroshot-eval2=meta-files/zs/HAM-official-7-zero-shot-test.csv \
    --zeroshot-eval3=meta-files/zs/f17k-113-zero-shot-test.csv \
    --zeroshot-eval4=meta-files/zs/daffodil-5-zero-shot-test.csv \
    --csv-label-key label \
    --csv-img-key image_path \
    --csv-caption-key 'truncated_caption' \
    --model 'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256'

# MAKE
python src/main.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size=1024 \
    --zeroshot-eval1=meta-files/zs/pad-zero-shot-test.csv \
    --zeroshot-eval2=meta-files/zs/HAM-official-7-zero-shot-test.csv \
    --zeroshot-eval3=meta-files/zs/f17k-113-zero-shot-test.csv \
    --zeroshot-eval4=meta-files/zs/daffodil-5-zero-shot-test.csv \
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
    --zeroshot-eval1=meta-files/zs/pad-zero-shot-test.csv \
    --zeroshot-eval2=meta-files/zs/HAM-official-7-zero-shot-test.csv \
    --zeroshot-eval3=meta-files/zs/f17k-113-zero-shot-test.csv \
    --zeroshot-eval4=meta-files/zs/daffodil-5-zero-shot-test.csv \
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
    --zeroshot-eval1=meta-files/zs/pad-zero-shot-test.csv \
    --zeroshot-eval2=meta-files/zs/HAM-official-7-zero-shot-test.csv \
    --zeroshot-eval3=meta-files/zs/f17k-113-zero-shot-test.csv \
    --zeroshot-eval4=meta-files/zs/daffodil-5-zero-shot-test.csv \
    --csv-label-key label \
    --csv-img-key image_path \
    --csv-caption-key 'truncated_caption' \
    --model 'hf-hub:ryanyip7777/pmc_vit_l_14'

# BiomedCLIP
python src/main.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size=1024 \
    --zeroshot-eval1=meta-files/zs/pad-zero-shot-test.csv \
    --zeroshot-eval2=meta-files/zs/HAM-official-7-zero-shot-test.csv \
    --zeroshot-eval3=meta-files/zs/f17k-113-zero-shot-test.csv \
    --zeroshot-eval4=meta-files/zs/daffodil-5-zero-shot-test.csv \
    --csv-label-key label \
    --csv-img-key image_path \
    --csv-caption-key 'truncated_caption' \
    --model 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
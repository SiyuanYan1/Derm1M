# DermLIP - ViT-B-16
## Hold Out Dataset
python src/main.py \
    --val-data="data/pretrain/derm1m-valid.csv"  \
    --csv-caption-key 'truncated_caption' \
    --dataset-type "csv" \
    --batch-size=2048 \
    --csv-label-key label \
    --csv-img-key filename \
    --model 'hf-hub:redlessone/DermLIP_ViT-B-16'

## SkinCAP
python src/main.py \
    --val-data="data/meta/skincap.csv"  \
    --csv-caption-key 'caption_zh_polish_en' \
    --dataset-type "csv" \
    --batch-size=2048 \
    --csv-label-key label \
    --csv-img-key filename \
    --model 'hf-hub:redlessone/DermLIP_ViT-B-16'

# DermLIP - ViT-B-16
## Hold Out Dataset
python src/main.py \
    --val-data="data/pretrain/derm1m-valid.csv"  \
    --csv-caption-key 'truncated_caption' \
    --dataset-type "csv" \
    --batch-size=2048 \
    --csv-label-key label \
    --csv-img-key filename \
    --model 'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256'

## SkinCAP
python src/main.py \
    --val-data="data/meta/skincap.csv"  \
    --csv-caption-key 'caption_zh_polish_en' \
    --dataset-type "csv" \
    --batch-size=2048 \
    --csv-label-key label \
    --csv-img-key filename \
    --model 'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256'

# MAKE
## Hold Out Dataset
python src/main.py \
    --val-data="data/pretrain/derm1m-valid.csv"  \
    --csv-caption-key 'truncated_caption' \
    --dataset-type "csv" \
    --batch-size=2048 \
    --csv-label-key label \
    --csv-img-key filename \
    --model 'hf-hub:xieji-x/MAKE'

## SkinCAP
python src/main.py \
    --val-data="data/meta/skincap.csv"  \
    --csv-caption-key 'caption_zh_polish_en' \
    --dataset-type "csv" \
    --batch-size=2048 \
    --csv-label-key label \
    --csv-img-key filename \
    --model 'hf-hub:xieji-x/MAKE'
# DermLIP: PanDerm-base-w-PubMed-256
python concept_annotation/automatic_concept_annotation.py \
    --data_dir "meta-files/concept_annotation/skincon" \
    --batch_size 32 \
    --concept_list "meta-files/concept_annotation/skincon/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json" \
    --model_api open_clip_hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256

python concept_annotation/automatic_concept_annotation.py \
    --data_dir "meta-files/concept_annotation/derm7pt" \
    --batch_size 32 \
    --concept_list "data/derm7pt/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json"
    --model_api open_clip_hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256

# DermLIP: DermLIP_ViT-B-16
python concept_annotation/automatic_concept_annotation.py \
    --data_dir "meta-files/concept_annotation/skincon" \
    --batch_size 32 \
    --concept_list "meta-files/concept_annotation/skincon/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json" \
    --model_api open_clip_hf-hub:redlessone/DermLIP_ViT-B-16

python concept_annotation/automatic_concept_annotation.py \
    --data_dir "meta-files/concept_annotation/derm7pt" \
    --batch_size 32 \
    --concept_list "data/derm7pt/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json"
    --model_api open_clip_hf-hub:redlessone/DermLIP_ViT-B-16

# MAKE
python concept_annotation/automatic_concept_annotation.py \
    --data_dir "meta-files/concept_annotation/skincon" \
    --batch_size 32 \
    --concept_list "meta-files/concept_annotation/skincon/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json" \
    --model_api open_clip_hf-hub:xieji-x/MAKE

python concept_annotation/automatic_concept_annotation.py \
    --data_dir "meta-files/concept_annotation/derm7pt" \
    --batch_size 32 \
    --concept_list "data/derm7pt/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json"
    --model_api open_clip_hf-hub:xieji-x/MAKE
#!/bin/bash
rm -r linear_probing_logs/*
models=('open_clip_hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256' 'open_clip_hf-hub:redlessone/DermLIP_ViT-B-16')
datasets=('HAM' 'PAD' 'f17k-113' 'Daffodil')
percent_data=1
csv_paths=(
  '../meta-files/lp/HAM-official-7-lp.csv'
  '../meta-files/lp/pad-lp-ws0.csv'
  '../meta-files/lp/f17k-113-lp-ws0.csv'
  '../meta-files/lp/daffodil-5-lp.csv'
)

for model in "${models[@]}"; do
  for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    csv_path="${csv_paths[$i]}"
    
    output_dir="linear_probing_logs/${model}-${dataset}/pretrain"
    
    if [ ! -f "$csv_path" ]; then
      echo "CSV file not found: $csv_path"
      continue
    fi
    
    csv_filename="${model}-${dataset}_results.csv"
    echo "Processing model: ${model}, dataset: ${dataset}"
    
    CUDA_VISIBLE_DEVICES=0 python linear_eval.py \
      --batch_size 1000 \
      --model "${model}" \
      --csv_filename "${csv_filename}" \
      --output_dir "${output_dir}" \
      --csv_path "${csv_path}" \
      --image_key "image_path" \
      --percent_data ${percent_data}
  done
  wait
done

python sort_script.py
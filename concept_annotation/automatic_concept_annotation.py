import pyrootutils
import sys
import os

project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from src import open_clip
from concept_annotation.utils import load_concept_list, concept_mapping, concept_prompt_template_list, concept_prompt_ref_list
from concept_annotation.dataset import *
from concept_annotation.model import *

import torch

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import glob
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import argparse

def batch_func(batch):
    with torch.no_grad():
        image_features = model.encode_image(batch["image"].to(device))

    return {
        "image_features": image_features.detach().cpu(),
        "metadata": batch["metadata"],
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run concept annotation model.')
    parser.add_argument('--model_api', type=str, default='clip',
                        help='Choose the model API to use: clip or monet.')
    parser.add_argument('--data_dir', type=str, default='data/concept-annotation/skincon',
                        help='Directory containing the images and metadata.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing images.')
    parser.add_argument('--concept_list', type=str, default=None,
                        help='Path to a file containing concept terms or a comma-separated list of concepts.')
    parser.add_argument('--concept_terms_json', type=str, default='concept_annotation/term_lists/ConceptTerms.json',
                        help='Path to the JSON file containing the mapping of concept terms.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to the model weight')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    model_api = args.model_api
    dataset_dir = args.data_dir
    batch_size = args.batch_size
    concept_list = load_concept_list(args.concept_list)
    print(concept_list)
    concept_terms_json_path = args.concept_terms_json

    # Model initilaize
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_api.startswith('open_clip_'):
        model, _, preprocess = open_clip.create_model_and_transforms(model_api.replace('open_clip_', ''))
        
        model = model.to(device)
        model.eval()
    
    if args.resume is not None:
        state_dict = torch.load(args.resume, weights_only=False)['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Load pretrained weight from {args.resume}")

    # specify the directory containing the images
    image_path_list = [Path(path) for path in glob.glob(str(Path(dataset_dir + '/final_images') / "*"))]
    image_path_list = [
        image_path
        for image_path in image_path_list
        if image_path.suffix in [".png", ".jpg", ".jpeg", '.JPG']
    ]

    image_dataset = ImageDataset(
        image_path_list, preprocess, metadata_df=None
    )

    gt_df = pd.read_csv(dataset_dir + '/meta.csv')
    image_list = [str(i).split('/')[-1] for i in image_path_list]
    order_df = pd.DataFrame(image_list, columns=['ImageID'])

    # Merge the order_df with gt_df to reorder rows according to image_list
    gt_df = order_df.merge(gt_df, on='ImageID', how='left')
    print(gt_df)

    dataloader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=custom_collate,
        shuffle=False,
    )

    image_embedding = dataloader_apply_func(
        dataloader=dataloader,
        func=batch_func,
        collate_fn=custom_collate_per_key,
    )

    filtered_concept_terms_dict = concept_mapping(concept_list, concept_terms_json_path)
    concept_embedding_dict = {}

    for key in filtered_concept_terms_dict.keys():
        concept_list = filtered_concept_terms_dict[key]
        concept_embedding = get_prompt_embedding(model, concept_term_list=concept_list, 
                                                 prompt_template_list=concept_prompt_template_list, prompt_ref_list=concept_prompt_ref_list, model_api = model_api if model_api.startswith('open_clip') else None)
        concept_embedding_dict[key] = concept_embedding


    image_features_norm = image_embedding["image_features"] / image_embedding[
        "image_features"
    ].norm(dim=1, keepdim=True)

    concept_presence_score_dict = {}

    for concept in concept_embedding_dict.keys():
        concept_presence_score = calculate_concept_presence_score(
            image_features_norm=image_features_norm,
            prompt_target_embedding_norm=concept_embedding_dict[concept]['prompt_target_embedding_norm'],
            prompt_ref_embedding_norm=concept_embedding_dict[concept]['prompt_ref_embedding_norm'],
        )
        concept_presence_score_dict[concept] = concept_presence_score

    concept_list = list(concept_presence_score_dict.keys())
    presence_score = [concept_presence_score_dict[v] for v in concept_list]
    presence_score = np.stack(presence_score).T

    # build GT vector according to Dataframe
    gt_results = []
    for i in range(len(gt_df)):
        result = []
        for j in gt_df.columns[1:]:
            result.append(gt_df.loc[i, j])
        gt_results.append(result)

    gt_results = np.stack(gt_results) # export GT with a matrix

    roc_aucs = []
    # Compute ROC AUC for each class
    for i in range(gt_results.shape[1]):  # assuming gt_results and presence_score have the same number of columns
        # True labels for the current class
        true_labels = gt_results[:, i]
        # Scores or probabilities for the current class
        scores = presence_score[:, i]

        # Calculate ROC AUC for the current class
        roc_auc = roc_auc_score(true_labels, scores)
        roc_aucs.append(roc_auc)
        print(f"ROC AUC for class {i} - {concept_list[i]}: {roc_auc}")

    # Optional: Convert list to a numpy array if you want to manipulate or analyze the AUC scores further
    roc_aucs = np.array(roc_aucs)

    print(f"overall AUROC: {roc_aucs.mean()}({len([r for r in roc_aucs if r > 0.7])}/{len(roc_aucs)})")
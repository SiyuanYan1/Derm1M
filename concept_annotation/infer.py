import pyrootutils
import sys
import os

project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from concept_annotation.utils import load_concept_list, concept_mapping, concept_prompt_template_list, concept_prompt_ref_list
import torch
import numpy as np
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from concept_annotation.dataset import *
from concept_annotation.model import *
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
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing images.')
    parser.add_argument('--concept_list', type=str, default=None,
                        help='Path to a file containing concept terms or a comma-separated list of concepts.')
    parser.add_argument('--concept_terms_json', type=str, default='/home/ander/repos/external/MONET/ConceptTerms.json',
                        help='Path to the JSON file containing the mapping of concept terms.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the model weight')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    model_api = args.model_api
    image_path = args.image_path
    batch_size = args.batch_size
    concept_list = load_concept_list(args.concept_list)
    concept_terms_json_path = args.concept_terms_json

    # Model initilaize
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_api.startswith('open_clip_'):
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_api.replace('open_clip_', ''))
        
        model = model.to(device)
        model.eval()
    
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Load pretrained weight from {args.checkpoint}")

    # Process single image
    image_path_list = [Path(image_path)]

    image_dataset = ImageDataset(
        image_path_list, preprocess, metadata_df=None
    )

    dataloader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=1,
        num_workers=1,
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

    concept_names = list(concept_presence_score_dict.keys())
    presence_scores = [concept_presence_score_dict[v][0] for v in concept_names]  # Get score for single image
    
    # Get top 5 concepts
    top5_indices = np.argsort(presence_scores)[::-1][:5]
    
    print("Top 5 skin concepts:")
    for i, idx in enumerate(top5_indices, 1):
        print(f"{i}. {concept_names[idx]}: {presence_scores[idx]:.4f}")
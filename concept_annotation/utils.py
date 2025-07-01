import os
import json
import torch
import torchvision.transforms as T
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm

def custom_collate(batch: List[Dict]) -> Dict:
    """Custom collate function for the dataloader.

    Args:
        batch (List[Dict]): List of dictionaries, each dictionary is a batch of data.

    Returns:
        Dict: Dictionary of collated data.
    """

    ret = {}
    for key in batch[0]:
        if isinstance(batch[0][key], pd.Series):
            try:
                ret[key] = pd.concat([d[key] for d in batch], axis=1).T
            except RuntimeError:
                raise RuntimeError(f"Error while concatenating {key}")
        else:
            try:
                ret[key] = torch.utils.data.dataloader.default_collate(
                    [d[key] for d in batch]
                )
            except RuntimeError:
                raise RuntimeError(f"Error while concatenating {key}")

    return ret


def custom_collate_per_key(batch_all: Dict) -> Dict:
    """Custom collate function for batched outputs.

    Args:
        batch_all (Dict): Dictionary of lists of objects, each dictionary is a batch of data.

    Returns:
        Dict: Dictionary of collated data.
    """

    ret = {}
    for key in batch_all:
        if isinstance(batch_all[key][0], pd.DataFrame):
            ret[key] = pd.concat(batch_all[key], axis=0)
        elif isinstance(batch_all[key][0], torch.Tensor):
            ret[key] = torch.concat(batch_all[key], axis=0)
        else:
            print(f"Collating {key}...")
            ret[key] = torch.utils.data.dataloader.default_collate(
                [elem for batch in tqdm(batch_all[key]) for elem in batch]
            )

    return ret


def dataloader_apply_func(
    dataloader, func, collate_fn=custom_collate_per_key, verbose=True
):
    """Apply a function to a dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): Torch dataloader.
        func (function): Function to apply to each batch.
        collate_fn (function, optional): Collate function. Defaults to custom_collate_per_key.

    Returns:
        Dict: Dictionary of outputs.
    """
    func_out_dict = {}

    for batch in dataloader:
        for key, func_out in func(batch).items():
            func_out_dict.setdefault(key, []).append(func_out)

    return collate_fn(func_out_dict)


def get_transform(n_px: int):
    """Get image transformations.

    Args:
        n_px (int): Target image size.

    Returns:
        torchvision.transforms.Compose: Composed image transformations.
    """

    def convert_image_to_rgb(image):
        return image.convert("RGB")

    return T.Compose(
        [
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(n_px),
            convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

def similarity(im_features, text_features):
    sim = text_features.float() @ im_features.T.float()
    return sim

def load_concept_list(concept_list_arg):
    if os.path.exists(concept_list_arg):
        # Assume it's a file path
        with open(concept_list_arg, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    else:
        # Assume it's a comma-separated list
        return concept_list_arg.split(',')

def concept_mapping(concept_list, concept_term_path):
    concept_list = [s.lower() for s in concept_list]
    # Read concept term from MONET implementary
    with open(concept_term_path, 'r') as file:
        concept_terms_dict = json.load(file)

    return {key: concept_terms_dict[key] for key in concept_list if key in list(concept_terms_dict.keys())}

# --- concept annotation template and refernece list
OPENAI_SKIN_TEMPLATES = (
    lambda c: f'This is a skin image of {c}',
    lambda c: f'A skin image of {c}.',
    lambda c: f'An image of {c}, a skin condition.',
    lambda c: f'{c}, a skin disorder, is shown in this image.',
    lambda c: f'The skin lesion depicted is {c}.',
    lambda c: f'The skin cancer in this image is {c}.',
    lambda c: f'This image depicts {c}, a type of skin cancer.',
)

concept_prompt_template_list=[
    "This is skin image of {}",
    "This is dermatology image of {}",
    "This is image of {}",
    "A skin image of {}", 
    "An image of {}, a skin condition.",
    "{}, a skin disorder, is shown in this image."
]

concept_prompt_ref_list=[
    ["This is skin image"],
    ["This is dermatology image"],
    ["This is image"],
    ["A skin image"], 
    ["An image of a skin condition."],
    ["a skin disorder is shown in this image."]
]

disease_prompt_template_list=[
    "This is skin image of {}",
    "This is dermatology image of {}",
    "This is image of {}",
    "A skin image of {}", 
    "An image of {}, a skin condition.",
    "{}, a skin disorder, is shown in this image.",
    "The skin lesion depicted is {}.",
    "The skin cancer in this image is {}.",
    "This image depicts {}, a type of skin cancer."
]

disease_prompt_ref_list=[
    ["This is skin image"],
    ["This is dermatology image"],
    ["This is image"],
    ["A skin image"], 
    ["An image of a skin condition."],
    ["a skin disorder is shown in this image."],
    ["The skin leision is depicted."],
    ["The skin cancer is in this image."],
    ["This image depicts a type of skin cancner."]
]
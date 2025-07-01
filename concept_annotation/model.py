import torch
import torchvision.transforms as T

import numpy as np
import scipy

OPENAI_SKIN_TEMPLATES = (
    lambda c: f'This is a skin image of {c}',
    lambda c: f'A skin image of {c}.',
    lambda c: f'An image of {c}, a skin condition.',
    lambda c: f'{c}, a skin disorder, is shown in this image.',
    lambda c: f'The skin lesion depicted is {c}.',
    lambda c: f'The skin cancer in this image is {c}.',
    lambda c: f'This image depicts {c}, a type of skin cancer.',
)

def get_prompt_embedding(
    model,
    concept_term_list=[],
    prompt_template_list=[
        "This is skin image of {}",
        "This is dermatology image of {}",
        "This is image of {}",
        "A skin image of {}", 
        "An image of {}, a skin condition.",
        "{}, a skin disorder, is shown in this image."
    ],
    prompt_ref_list=[
        ["This is skin image"],
        ["This is dermatology image"],
        ["This is image"],
        ["A skin image"], 
        ["An image of a skin condition."],
        ["a skin disorder is shown in this image."]
    ],
    model_api = None
):
    """
    Generate prompt embeddings for a concept

    Args:
        concept_term_list (list): List of concept terms that will be used to generate prompt target embeddings.
        prompt_template_list (list): List of prompt templates.
        prompt_ref_list (list): List of reference phrases.

    Returns:
        dict: A dictionary containing the normalized prompt target embeddings and prompt reference embeddings.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_api != None and 'open_clip' in model_api:
        import open_clip
        tokenizer = open_clip.factory.get_tokenizer(model_api.replace('open_clip_',''))

    # target embedding
    prompt_target = [
        [prompt_template.format(term) for term in concept_term_list]
        for prompt_template in prompt_template_list
    ]

    prompt_target_tokenized = [
        tokenizer(prompt_list).to(device) for prompt_list in prompt_target
    ]
    with torch.no_grad():
        prompt_target_embedding = torch.stack(
            [
                model.encode_text(prompt_tokenized.to(next(model.parameters()).device)).detach().cpu()
                for prompt_tokenized in prompt_target_tokenized
            ]
        )
    prompt_target_embedding_norm = (
        prompt_target_embedding / prompt_target_embedding.norm(dim=2, keepdim=True)
    )

    # reference embedding
    prompt_ref_tokenized = [
        tokenizer(prompt_list) for prompt_list in prompt_ref_list
    ]
    with torch.no_grad():
        prompt_ref_embedding = torch.stack(
            [
                model.encode_text(prompt_tokenized.to(next(model.parameters()).device)).detach().cpu()
                for prompt_tokenized in prompt_ref_tokenized
            ]
        )
    prompt_ref_embedding_norm = prompt_ref_embedding / prompt_ref_embedding.norm(
        dim=2, keepdim=True
    )

    return {
        "prompt_target_embedding_norm": prompt_target_embedding_norm,
        "prompt_ref_embedding_norm": prompt_ref_embedding_norm,
    }

# Calculate concept presence score
def calculate_concept_presence_score(
    image_features_norm,
    prompt_target_embedding_norm,
    prompt_ref_embedding_norm,
    temp=1 / np.exp(4.5944),
):
    """
    Calculates the concept presence score based on the given image features and concept embeddings.

    Args:
        image_features_norm (numpy.Tensor): Normalized image features.
        prompt_target_embedding_norm (torch.Tensor): Normalized concept target embedding.
        prompt_ref_embedding_norm (torch.Tensor): Normalized concept reference embedding.
        temp (float, optional): Temperature parameter for softmax. Defaults to 1 / np.exp(4.5944).

    Returns:
        np.array: Concept presence score.
    """

    target_similarity = (
        prompt_target_embedding_norm.float() @ image_features_norm.T.float()
    )
    ref_similarity = prompt_ref_embedding_norm.float() @ image_features_norm.T.float()

    target_similarity_mean = target_similarity.mean(dim=[1])
    ref_similarity_mean = ref_similarity.mean(axis=1)

    concept_presence_score = scipy.special.softmax(
        [target_similarity_mean.numpy() / temp, ref_similarity_mean.numpy() / temp],
        axis=0,
    )[0, :].mean(axis=0)

    return concept_presence_score
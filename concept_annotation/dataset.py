import torch
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tqdm
import warnings
warnings.filterwarnings("ignore")

def get_transform(n_px):
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

class ImageDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for loading images and associated metadata.

    Args:
        image_path_list (list): A list of file paths to the images.
        transform (callable): A function/transform to apply to the images.
        metadata_df (pandas.DataFrame, optional): A pandas DataFrame containing metadata for the images.

    Raises:
        AssertionError: If the length of `image_path_list` is not equal to the length of `metadata_df`.

    Returns:
        dict: A dictionary containing the image and metadata (if available) for a given index.

    """

    def __init__(self, image_path_list, transform, metadata_df=None):
        self.image_path_list = image_path_list
        self.transform = transform
        self.metadata_df = metadata_df

        if self.metadata_df is None:
            self.metadata_df = pd.Series(index=self.image_path_list)
        else:
            assert len(self.image_path_list) == len(
                self.metadata_df
            ), "image_path_list and metadata_df must have the same length"
            self.metadata_df.index = self.image_path_list

    def __getitem__(self, idx):
        image = Image.open(self.image_path_list[idx])

        ret = {"image": self.transform(image)}

        if self.metadata_df is not None:
            ret.update({"metadata": self.metadata_df.iloc[idx]})

        return ret

    def __len__(self):
        return len(self.image_path_list)

def custom_collate(batch):
    """Custom collate function for the dataloader.

    Args:
        batch (list): list of dictionaries, each dictionary is a batch of data

    Returns:
        dict: dictionary of collated data
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

def custom_collate_per_key(batch_all):
    """Custom collate function batched outputs.

    Args:
        batch_all (dict): dictionary of lists of objects, each dictionary is a batch of data
    Returns:
        dict: dictionary of collated data
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
                [elem for batch in tqdm.tqdm(batch_all[key]) for elem in batch]
            )

    return ret

def dataloader_apply_func(
    dataloader, func, collate_fn=custom_collate_per_key, verbose=True
):
    """Apply a function to a dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): torch dataloader
        func (function): function to apply to each batch
        collate_fn (function, optional): collate function. Defaults to custom_collate_batch.

    Returns:
        dict: dictionary of outputs
    """
    func_out_dict = {}

    for batch in tqdm.tqdm(dataloader):
        for key, func_out in func(batch).items():
            func_out_dict.setdefault(key, []).append(func_out)

    return collate_fn(func_out_dict)

class SkinLesionDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Load image
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get label
        label = 1 if row['diagnosis'] == 'malignant' else 0

        # Metadata
        metadata = row.to_dict()

        return {'image': image, 'label': label, 'metadata': metadata}

#
"""
Based on evaluate_zeroshot from SLIP but changed by MB.

Adapated from https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/linear_probe.py
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Tuple, Dict, Any, List
from warnings import simplefilter

import torch
import torch.utils.data
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from .logistic_regression import LogisticRegression
from .metrics import get_eval_metrics, get_eval_metrics_stats


# Silence repeated convergence warnings from scikit-learn logistic regression.
simplefilter("ignore", category=ConvergenceWarning)


def eval_linear_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    valid_feats: torch.Tensor,
    valid_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    test_filenames:List[str],
    max_iter: int = 1000,
    combine_trainval: bool = True,
    use_sklearn: bool = False,
    prefix: str = "lin_",
    verbose: bool = True, seed: int = 100,
    out_dir=None,
    dataset_name=None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Wrapper function that calls "train_linear_probe" and "test_linear_probe".

    Args:
        train_feats: The features of the training set.
        train_labels: The labels of the training set.
        valid_feats: The features of the validation set.
        valid_labels: The labels of the validation set.
        test_feats: The features of the test set.
        test_labels: The labels of the test set.
        use_mean_accuracy: Whether to compute mean accuracy.
        cost_search: Whether to perform cost hyperparameter search.
        sk_verbose: Whether to print verbose output from scikit-learn.
        max_iter: The maximum number of iterations for training the classifier.
        combine_trainval: Whether to combine the training and validation sets.
        use_sklearn: Whether to use scikit-learn's LogisticRegression.
        prefix: The prefix to use for the evaluation results.
        verbose: Whether to print verbose output.

    Returns:
        A tuple containing results (dict of eval metric name to value) and dump (dict of prob logits)
    """

    if verbose:
        print("Linear Probe Evaluation: Train shape", train_feats.shape)
    if valid_feats is not None:
        if verbose:
            print("Linear Probe Evaluation: Valid shape", valid_feats.shape)
    if verbose:
        print("Linear Probe Evaluation: Test shape", test_feats.shape)
    start = time.time()
    classifier = train_linear_probe(
        train_feats,
        train_labels,
        valid_feats,
        valid_labels,
        max_iter=max_iter,
        combine_trainval=combine_trainval,
        use_sklearn=use_sklearn,
        verbose=verbose, seed = seed
    )
    results, dump = test_linear_probe(classifier, test_feats, test_labels, test_filenames,prefix=prefix, verbose=verbose, out_dir=out_dir,dataset_name=dataset_name)
    classifier.logreg = classifier.logreg.to(torch.device("cpu"))
    dump["logreg"] = classifier.logreg.state_dict()
    del classifier
    torch.cuda.empty_cache()
    if verbose:
        print(f"Linear Probe Evaluation: Time taken {time.time() - start:.2f}")

    return results, dump

# import torchsnooper
# @torchsnooper.snoop()
def train_linear_probe(
    train_feats,
    train_labels,
    valid_feats,
    valid_labels,
    max_iter=1000,
    combine_trainval=True,
    use_sklearn=False,
    verbose=True, seed = 100
):
    """
    Args:
        holdout_fraction: Fraction of the (official) train split to hold out for
            validation while searching the cost hyperparameter. Holding out will
            be deterministic and have similar class distribution as train split.
    """
    NUM_C = len(set(train_labels.cpu().numpy()))
    cost = (train_feats.shape[1] * NUM_C) / 100
    # cost = 1.0
    print('NUM_C, Cost:',NUM_C,cost)
    if verbose:
        print(f"Linear Probe Evaluation (Train Time): Best cost = {cost:.3f}")

    # train final classifier
    if combine_trainval and (valid_feats is not None):
        trainval_feats = torch.cat([train_feats, valid_feats], dim=0)
        trainval_labels = torch.cat([train_labels, valid_labels], dim=0)
        if verbose:
            print("Linear Probe Evaluation (Train Time): Combining train and validation sets for final training. Trainval Shape: ", trainval_feats.shape)

        final_classifier = _fit_logreg(
            trainval_feats,
            trainval_labels,
            cost,
            verbose,
            max_iter,
            use_sklearn, seed 
        )
    else:
        if verbose:
            print("Linear Probe Evaluation (Train Time): Using only train set for training. Train Shape: ", train_feats.shape)

        final_classifier = _fit_logreg(
            train_feats, 
            train_labels, 
            cost, 
            verbose, 
            max_iter, 
            use_sklearn, seed
        )
        

    return final_classifier


import pandas as pd
import os


def save_predictions_csv(
        original_csv_path: str,
        predictions: np.ndarray,
        output_csv_path: str
):
    """
    Save predictions to a new CSV file with patient_id, timepoint, and prediction.

    Args:
        original_csv_path (str): Path to the original fold_1.csv file
        predictions (np.ndarray): Array of predictions (0 or 1) from the model
        output_csv_path (str): Path to save the new CSV file
    """
    # Read the original CSV file
    original_df = pd.read_csv(original_csv_path)

    # Filter only the test set
    test_df = original_df[original_df['split'] == 'test'].copy()

    # Ensure the number of predictions matches the number of test samples
    assert len(predictions) == len(test_df), "Number of predictions doesn't match number of test samples"

    # Add predictions to the dataframe
    test_df['pred'] = predictions

    # Extract timepoint (image name) from the image path
    test_df['timepoint'] = test_df['image'].apply(lambda x: os.path.basename(x))

    # Select and reorder columns
    result_df = test_df[['patient_id', 'timepoint', 'pred']]

    # Save to CSV
    result_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


# Example usage:
# Assuming you have run the test_linear_probe function and have the predictions

# original_csv_path = 'path/to/fold_1.csv'
# predictions = eval_metrics['preds_all']  # This should be the array of predictions from test_linear_probe
# output_csv_path = 'path/to/predictions_fold_1.csv'

# save_predictions_csv(original_csv_path, predictions, output_csv_path)

def test_linear_probe(
    linear_classifier: LogisticRegression,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    test_filenames:List[str],
    num_classes: int = None,
    prefix: str = "lin_",
    verbose: bool = True,
    out_dir: str=None,
    dataset_name: str=None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate the linear probe on the test set.

    Args:
        linear_classifier: The trained linear classifier.
        test_feats: The features of the test set.
        test_labels: The labels of the test set.
        num_classes: The number of classes in the dataset.
        prefix: The prefix to use for the evaluation results.
        verbose: Whether to print verbose output.

    Returns:
        A tuple containing the evaluation results and additional information.
    """
    if verbose:
        print(f"Linear Probe Evaluation (Test Time): Test Shape {test_feats.shape}")

    # evaluate
    NUM_C = len(set(test_labels.cpu().numpy())) if num_classes is None else num_classes

    if NUM_C == 2:
        probs_all = linear_classifier.predict_proba(test_feats)[:, 1].detach().cpu().numpy()
        roc_kwargs = {}
    else:
        probs_all = linear_classifier.predict_proba(test_feats).detach().cpu().numpy()
        roc_kwargs = {"multi_class": "ovo", "average": "macro"}

    preds_all = linear_classifier.predict_proba(test_feats).argmax(dim=1).detach().cpu().numpy()
    targets_all = test_labels.detach().cpu().numpy()
    import sklearn
    classification_report = sklearn.metrics.classification_report(
        y_true=targets_all, y_pred=preds_all
    )
    print('Classification report:')
    print(classification_report)
    eval_metrics = get_eval_metrics(out_dir,test_filenames,dataset_name,targets_all, preds_all, probs_all, True, prefix, roc_kwargs)

    # eval_metrics = get_eval_metrics_stats(out_dir,dataset_name,targets_all, preds_all, probs_all, True, prefix, roc_kwargs)
    dump = {"preds_all": preds_all, "probs_all": probs_all, "targets_all": targets_all}



    # # Now use the new function to save predictions
    # original_csv_path =  '/home/share/Uni_Eval/SDDI/fold_3.csv'
    # predictions = dump['preds_all']  # or eval_metrics['preds_all'], depending on where you stored the predictions
    # output_csv_path = '/home/share/Uni_Eval/SDDI/predictions_fold_3.csv'
    #
    # # save_predictions_csv(original_csv_path, predictions, output_csv_path)

    return eval_metrics, dump


def _fit_logreg(
    feats: torch.Tensor,
    labels: torch.Tensor,
    cost: float,
    verbose: bool = False,
    max_iter: int = 100,
    use_sklearn: bool = False, seed: int = 100,
) -> LogisticRegression:
    """
    Initialize and fit a `LogisticRegression` classifier for input features and
    labels. Default settings follow CLIP (L-BFGS, 1K iterations, etc.).

    Args:
        feats (torch.Tensor): Input features.
        labels (torch.Tensor): Input labels.
        cost (float): Inverse of regularization strength; smaller values specify stronger regularization.
        verbose (bool, optional): Whether to enable verbose output. Defaults to False.
        max_iter (int, optional): Maximum number of iterations taken for the solvers to converge. Defaults to 100.
        use_sklearn (bool, optional): Whether to use scikit-learn's LogisticRegression implementation. Defaults to False.

    Returns:
        LogisticRegression: Fitted logistic regression classifier.

    """
    if use_sklearn:
        classifier = sk_LogisticRegression(C=cost, max_iter=max_iter, verbose=verbose, random_state=seed)
    else:
        classifier = LogisticRegression(C=cost, max_iter=max_iter, verbose=verbose, random_state=seed)
    classifier.fit(feats, labels)
    return classifier


def split_trainval(
        targets: List[int], 
        val_percentage: float
) -> Dict[List[int], List[int]]:
    """
    Split the dataset into training and validation sets based on the given validation percentage.

    Args:
        targets: List of target labels.
        val_percentage: Percentage of data to be used for validation.

    Returns:
        A dictionary containing the indices of training and validation samples.
    """
    # Organize dataset by classes (class ID -> list[dataset index] map).
    labels_to_indices = defaultdict(list)
    for index, label in enumerate(targets):
        labels_to_indices[label].append(index)

    train_indices = []
    valid_indices = []
    for label, indices in labels_to_indices.items():
        # Deterministic shuffling to ensure same held-out split across runs.
        random.Random(93).shuffle(indices)

        train_indices.extend(indices[int(len(indices) * val_percentage) :])
        valid_indices.extend(indices[: int(len(indices) * val_percentage)])

    return train_indices, valid_indices

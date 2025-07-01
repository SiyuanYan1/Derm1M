from typing import Optional, Dict, Any, Union, List
import numpy as np

from pycm import *
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    average_precision_score
)

def get_eval_metrics_stats(
        out_dir: str,
        dataset_name: str,
        targets_all: Union[List[int], np.ndarray],
        preds_all: Union[List[int], np.ndarray],
        probs_all: Optional[Union[List[float], np.ndarray]] = None,
        get_report: bool = True,
        prefix: str = "",
        roc_kwargs: Dict[str, Any] = {},
        n_samples: int = 1000
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics with resampling and return the evaluation metrics.
    """
    import numpy as np

    import numpy as np

    def debug_class_mismatch(y_true, y_pred, y_prob):
        print("Shape of y_true:", y_true.shape)
        print("Shape of y_pred:", y_pred.shape)
        print("Shape of y_prob:", y_prob.shape)
        print("Unique classes in y_true:", np.unique(y_true))
        print("Unique classes in y_pred:", np.unique(y_pred))
        if y_prob.ndim > 1:
            print("Number of columns in y_prob:", y_prob.shape[1])
        else:
            print("y_prob is 1-dimensional (binary classification)")
        print("Data types - y_true:", y_true.dtype, "y_pred:", y_pred.dtype, "y_prob:", y_prob.dtype)

    # Add this before calling get_eval_metrics_stats
    debug_class_mismatch(targets_all, preds_all, probs_all)
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), np.std(a)
        h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, se, m - h, m + h
    # import torchsnooper
    # @torchsnooper.snoop()

    import numpy as np
    from sklearn.utils import resample

    import numpy as np
    from sklearn.utils import resample

    def resample_metric(metric_func, targets, preds, n_samples, is_prob=False):
        values = []
        unique_classes = np.unique(targets)
        n_classes = len(unique_classes)

        for _ in range(n_samples):
            resampled_targets = np.array([], dtype=int)
            resampled_preds = np.array([])

            # Resample for each class
            for class_label in unique_classes:
                class_indices = np.where(targets == class_label)[0]

                # Ensure at least one sample per class
                n_samples_class = max(1, len(class_indices))

                resampled_indices = resample(class_indices, n_samples=n_samples_class, replace=True)
                resampled_targets = np.concatenate([resampled_targets, targets[resampled_indices]])

                if is_prob and preds.ndim == 2:
                    if resampled_preds.size == 0:
                        resampled_preds = preds[resampled_indices]
                    else:
                        resampled_preds = np.vstack([resampled_preds, preds[resampled_indices]])
                else:
                    resampled_preds = np.concatenate([resampled_preds, preds[resampled_indices]])

            # Shuffle the resampled data
            shuffle_indices = np.arange(len(resampled_targets))
            np.random.shuffle(shuffle_indices)
            resampled_targets = resampled_targets[shuffle_indices]
            resampled_preds = resampled_preds[shuffle_indices]

            if is_prob:
                if resampled_preds.ndim == 2:
                    # For probability-based metrics (AUROC, AUPRC) in multi-class case
                    values.append(metric_func(resampled_targets, resampled_preds))
                else:
                    # For probability-based metrics (AUROC, AUPRC) in binary case
                    values.append(metric_func(resampled_targets, resampled_preds))
            else:
                # For other metrics
                values.append(metric_func(resampled_targets, resampled_preds))

        return mean_confidence_interval(values)
    targets_all = np.array(targets_all)
    preds_all = np.array(preds_all)
    eval_metrics = {}
    # Calculate metrics with resampling
    metrics = {
        f"{prefix}acc": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
        f"{prefix}bacc": lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred),
        f"{prefix}kappa": lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    for metric_name, metric_func in metrics.items():
        mean, std, ci_lower, ci_upper = resample_metric(metric_func, targets_all, preds_all, n_samples)
        eval_metrics[f"{metric_name}_mean"] = mean
        eval_metrics[f"{metric_name}_std"] = std
        eval_metrics[f"{metric_name}_ci"] = (ci_lower, ci_upper)

        # Resample weighted F1 score
        def weighted_f1(y_true, y_pred):
            return classification_report(y_true, y_pred, output_dict=True, zero_division=0)["weighted avg"]["f1-score"]

        mean, std, ci_lower, ci_upper = resample_metric(weighted_f1, targets_all, preds_all, n_samples)
        eval_metrics[f"{prefix}weighted_f1_mean"] = mean
        eval_metrics[f"{prefix}weighted_f1_std"] = std
        eval_metrics[f"{prefix}weighted_f1_ci"] = (ci_lower, ci_upper)
        # ROC AUC and PR AUC
    if probs_all is not None:
        probs_all = np.array(probs_all)

        def roc_auc_func(y_true, y_pred):
            return roc_auc_score(y_true, y_pred, **roc_kwargs)

        mean, std, ci_lower, ci_upper = resample_metric(roc_auc_func, targets_all, probs_all, n_samples, is_prob=True)
        eval_metrics[f"{prefix}auroc_mean"] = mean
        eval_metrics[f"{prefix}auroc_std"] = std
        eval_metrics[f"{prefix}auroc_ci"] = (ci_lower, ci_upper)

        def pr_auc_func(y_true, y_pred):
            return average_precision_score(y_true, y_pred, average='macro')

        mean, std, ci_lower, ci_upper = resample_metric(pr_auc_func, targets_all, probs_all, n_samples, is_prob=True)
        eval_metrics[f"{prefix}aupr_mean"] = mean
        eval_metrics[f"{prefix}aupr_std"] = std
        eval_metrics[f"{prefix}aupr_ci"] = (ci_lower, ci_upper)

    from sklearn.metrics import confusion_matrix
    cm_sum = np.zeros_like(confusion_matrix(targets_all, preds_all), dtype=float)
    successful_iterations=0
    for _ in range(n_samples):
        idx = np.random.randint(len(targets_all), size=len(targets_all))
        cm = confusion_matrix(targets_all[idx], preds_all[idx])
        try:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_sum += cm_normalized
            successful_iterations += 1
        except ValueError:
            # Skip this iteration if there's a mismatch error
            continue

    # Calculate mean only for successful iterations
    if successful_iterations > 0:
        cm_mean = cm_sum / successful_iterations
    else:
        cm_mean = np.zeros_like(cm_sum)  # Fallback if all iterations failed

    eval_metrics[f"{prefix}confusion_matrix_normalized"] = cm_mean

    print(f"Successful iterations: {successful_iterations} out of {n_samples}")

    # Save confusion matrix plot
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_mean, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(cm_mean.shape[0])
    plt.xticks(tick_marks, rotation=45)
    plt.yticks(tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add text annotations
    thresh = cm_mean.max() / 2.
    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            plt.text(j, i, f"{cm_mean[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm_mean[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'normalized_confusion_matrix_{dataset_name}.jpg'), dpi=600, bbox_inches='tight')
    plt.close()

    return eval_metrics
#
# def get_eval_metrics(
#     out_dir: None,
#     dataset_name: None,
#     targets_all: Union[List[int], np.ndarray],
#     preds_all: Union[List[int], np.ndarray],
#     probs_all: Optional[Union[List[float], np.ndarray]] = None,
#     get_report: bool = True,
#     prefix: str = "",
#     roc_kwargs: Dict[str, Any] = {},
# ) -> Dict[str, Any]:
#     """
#     Calculate evaluation metrics and return the evaluation metrics.
#     Args:
#         targets_all (array-like): True target values.
#         preds_all (array-like): Predicted target values.
#         probs_all (array-like, optional): Predicted probabilities for each class. Defaults to None.
#         get_report (bool, optional): Whether to include the classification report in the results. Defaults to True.
#         prefix (str, optional): Prefix to add to the result keys. Defaults to "".
#         roc_kwargs (dict, optional): Additional keyword arguments for calculating ROC AUC. Defaults to {}.
#     Returns:
#         dict: Dictionary containing the evaluation metrics.
#
#     """
#     bacc = balanced_accuracy_score(targets_all, preds_all)
#     kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
#     acc = accuracy_score(targets_all, preds_all)
#     cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)
#     cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)
#
#     eval_metrics = {
#         f"{prefix}acc": acc,
#         f"{prefix}bacc": bacc,
#         f"{prefix}kappa": kappa,
#         f"{prefix}weighted_f1": cls_rep["weighted avg"]["f1-score"],
#     }
#
#     if get_report:
#         eval_metrics[f"{prefix}report"] = cls_rep
#
#     if probs_all is not None:
#         roc_auc = roc_auc_score(targets_all, probs_all, **roc_kwargs)
#         eval_metrics[f"{prefix}auroc"] = roc_auc
#         auc_pr = average_precision_score(targets_all, probs_all, average='macro')
#         eval_metrics[f"{prefix}aupr"] = auc_pr
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     # Round the actual_vector and predict_vector to 4 decimal places
#     targets_all_rounded = np.round(targets_all, decimals=4)
#     preds_all_rounded = np.round(preds_all, decimals=4)
#
#     cm = ConfusionMatrix(targets_all_rounded, preds_all_rounded)
#
#     cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=False, plot_lib="matplotlib")
#
#     plt.savefig(out_dir + 'confusion_matrix_'+dataset_name+'.jpg', dpi=600, bbox_inches='tight')
#     plt.close()
#     return eval_metrics

def print_metrics(eval_metrics):
    for k, v in eval_metrics.items():
        if "report" in k:
            continue
        print(f"Test {k}: {v:.6f}")

def record_metrics_to_csv(eval_metrics, dataset_name, csv_filename, out_dir):
    def format_metric(value):
        return f"{value:.4f}"

    metrics = {
        "Dataset": dataset_name,
        "W_F1": format_metric(eval_metrics.get("lin_weighted_f1", 0)),
        "AUROC": format_metric(eval_metrics.get("lin_auroc", 0)),
        "BACC": format_metric(eval_metrics.get("lin_bacc", 0)),
        "ACC": format_metric(eval_metrics.get("lin_acc", 0)),
        "AUPR": format_metric(eval_metrics.get("lin_aupr", 0))
    }

    fieldnames = list(metrics.keys())
    row = list(metrics.values())

    csv_filepath = os.path.join(out_dir, csv_filename)
    os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow(row)

    print(f"Metrics recorded to {csv_filepath}")

import csv
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, accuracy_score, classification_report, \
    roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

def get_eval_metrics(
        out_dir: str,
        test_filenames: List[str],
        dataset_name: str,
        targets_all: Union[List[int], np.ndarray],
        preds_all: Union[List[int], np.ndarray],
        probs_all: Optional[Union[List[float], np.ndarray]] = None,
        get_report: bool = True,
        prefix: str = "",
        roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics and return the evaluation metrics.
    Also saves raw predictions and probabilities to a CSV file.
    """
    print(f"Debug: targets_all shape: {np.shape(targets_all)}")
    print(f"Debug: preds_all shape: {np.shape(preds_all)}")
    print(f"Debug: probs_all shape: {np.shape(probs_all) if probs_all is not None else 'None'}")
    print(f"Debug: test_filenames length: {len(test_filenames)}")

    bacc = balanced_accuracy_score(targets_all, preds_all)
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)
    eval_metrics = {
        f"{prefix}acc": acc,
        f"{prefix}bacc": bacc,
        f"{prefix}kappa": kappa,
        f"{prefix}weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }
    if get_report:
        eval_metrics[f"{prefix}report"] = cls_rep
    if probs_all is not None:
        try:
            roc_auc = roc_auc_score(targets_all, probs_all, **roc_kwargs)
            eval_metrics[f"{prefix}auroc"] = roc_auc
            auc_pr = average_precision_score(targets_all, probs_all, average='macro')
            eval_metrics[f"{prefix}aupr"] = auc_pr
        except Exception as e:
            print(f"Error calculating ROC AUC or PR AUC: {str(e)}")

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Save raw predictions and probabilities to CSV
    csv_filename = f"{dataset_name}.csv"
    csv_path = os.path.join(out_dir, csv_filename)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'true_label', 'predicted_label']
        if probs_all is not None:
            if isinstance(probs_all, np.ndarray) and probs_all.ndim > 1:
                fieldnames.extend([f'probability_class_{i}' for i in range(probs_all.shape[1])])
            else:
                fieldnames.append('probability')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, filename in enumerate(test_filenames):
            row = {
                'filename': filename,
                'true_label': targets_all[i],
                'predicted_label': preds_all[i]
            }
            if probs_all is not None:
                if isinstance(probs_all, np.ndarray) and probs_all.ndim > 1:
                    for j, prob in enumerate(probs_all[i]):
                        row[f'probability_class_{j}'] = prob
                else:
                    row['probability'] = probs_all[i]
            writer.writerow(row)
    print(f"Model predicted results saved to {csv_path}")

    # Create confusion matrix plot
    targets_all_rounded = np.round(targets_all, decimals=4)
    preds_all_rounded = np.round(preds_all, decimals=4)
    cm = confusion_matrix(targets_all_rounded, preds_all_rounded)
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(conf_mat=cm, figsize=(10, 8), show_absolute=True, show_normed=True, colorbar=True)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.savefig(os.path.join(out_dir, f'confusion_matrix_{dataset_name}.jpg'), dpi=600, bbox_inches='tight')
    plt.close()

    return eval_metrics


# Usage example:
# eval_metrics = get_eval_metrics(
#     out_dir="path/to/output/directory",
#     dataset_name="your_dataset_name",
#     model_name="your_model_name",
#     targets_all=your_targets,
#     preds_all=your_predictions,
#     probs_all=your_probabilities,
#     get_report=True,
#     prefix="lin_",
#     roc_kwargs={},
# )
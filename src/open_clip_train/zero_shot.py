import logging

import torch
from tqdm import tqdm

from open_clip_train.precision import get_autocast
from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    OPENAI_SKIN_TEMPLATES,HAM_CLASSNAMES,PAD_CLASSNAMES, DAFFODIL_5_CLASSNAMES, F17K_DISEASE_113_CLASSES

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, top_k_accuracy_score
import numpy as np
from torch import nn


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, num_class, args, metric='f1'):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        # Multi-class classification
        top1_correct = 0.0
        total_samples = 0.0
        true_labels_list = []
        prediction_labels_list = []
        targets_one_hot = []
        predictions_probs = []

        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)
            true_labels = target.to(torch.int64)

            with autocast():
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100.0 * image_features @ classifier
                prediction_softmax = torch.softmax(logits, dim=1)
                prediction_decode = prediction_softmax.argmax(dim=1)

            # Compute accuracy
            acc1 = accuracy(logits, true_labels, topk=(1,))
            batch_size = images.size(0)
            top1_correct += acc1[0] * batch_size / 100.0  # Convert percentage back to count
            total_samples += batch_size

            # Collect data for metrics
            true_labels_list.extend(true_labels.cpu().numpy())
            prediction_labels_list.extend(prediction_decode.cpu().numpy())
            targets_one_hot.extend(F.one_hot(true_labels, num_classes=num_class).cpu().numpy())
            predictions_probs.extend(prediction_softmax.cpu().numpy())

        # Compute metrics
        top1_acc = top1_correct / total_samples * 100.0  # Convert back to percentage
        true_labels_array = np.array(true_labels_list)
        prediction_labels_array = np.array(prediction_labels_list)
        targets_array = np.array(targets_one_hot)
        predictions_array = np.array(predictions_probs)

        # Compute weighted F1 and macro AUROC
        if metric == 'f1':
            auroc = roc_auc_score(targets_array, predictions_array, multi_class='ovr', average='macro')
            f1 = f1_score(true_labels_array, prediction_labels_array, average='weighted')

            return auroc, f1
        
        # Compute top1-acc and top5-acc
        elif metric == 'acc':
            top1_acc = accuracy_score(true_labels_array, prediction_labels_array)
            top5_acc = top_k_accuracy_score(true_labels_array, predictions_array, k=5)

            return top1_acc, top5_acc

        # Compute weighted F1 and top1-acc
        elif metric == 'f1+acc':
            top1_acc = accuracy_score(true_labels_array, prediction_labels_array)
            wf1 = f1_score(true_labels_array, prediction_labels_array, average='weighted')
            return top1_acc, wf1
        
        # Compute marco AUROC and acc
        elif metric == 'auroc+acc':
            auroc = roc_auc_score(targets_array, predictions_array, multi_class='ovr', average='macro')
            acc = accuracy_score(true_labels_array, prediction_labels_array)
            return auroc, acc


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    print('data source:',data)
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)

    templates=OPENAI_SKIN_TEMPLATES

    with autocast():
        if args.zeroshot_eval1:
            classifier_pad = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=PAD_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if args.zeroshot_eval2:
            classifier_ham = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=HAM_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if args.zeroshot_eval3:
            classifier_f17k_113_disease = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=F17K_DISEASE_113_CLASSES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if args.zeroshot_eval4:
            classifier_DAFFODIL_5 = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=DAFFODIL_5_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

    logging.info('Using classifier')
    results = {}
    # pad
    if args.zeroshot_eval1:
        test_auroc, acc = run(model, classifier_pad, data['zeroshot_pad'].dataloader, len(PAD_CLASSNAMES), args, metric='auroc+acc')
        results['zeroshot-pad-auroc'] = test_auroc
        results['zeroshot-pad-acc'] = acc

    # ham
    if args.zeroshot_eval2:
        auroc, acc = run(model, classifier_ham, data['zeroshot_ham'].dataloader, len(HAM_CLASSNAMES),args, metric='auroc+acc')
        results['zeroshot-ham-auroc'] = auroc
        results['zeroshot-ham-acc'] = acc

    # f17k - 113
    if args.zeroshot_eval3:
        top1_acc, top5_acc = run(model, classifier_f17k_113_disease, data['zeroshot_f17k_113_disease'].dataloader, len(F17K_DISEASE_113_CLASSES),args, metric='acc')
        results['zeroshot-f17k-113-top1-acc'] = top1_acc
        results['zeroshot-f17k-113-top5-acc'] = top5_acc

    # DAFFODIL - 5
    if args.zeroshot_eval4:
        auroc, acc = run(model, classifier_DAFFODIL_5, data['zeroshot_DAFFODIL-5-classes'].dataloader, len(DAFFODIL_5_CLASSNAMES),args, metric='auroc+acc')
        results['zeroshot-Daffodil-5-auroc'] = auroc
        results['zeroshot-Daffodil-5-acc'] = acc

    logging.info('Finished zero-shot.')
    return results
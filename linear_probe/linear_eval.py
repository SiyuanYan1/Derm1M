import torch
import pandas as pd
import argparse
import time

from datasets.derm_data import Derm_Dataset
from panderm_model.downstream.eval_features.metrics import get_eval_metrics, print_metrics,record_metrics_to_csv
from models import get_encoder
from panderm_model.downstream.extract_features import extract_features_from_dataloader
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_args_parser():
    parser = argparse.ArgumentParser('linear probing for skin image classification', add_help=False)
    parser.add_argument('--csv_path', default="/home/share/Uni_Eval/Derm7pt/atlas-clinical-all.csv" , type=str,
                    help='csv file path')
    parser.add_argument('--csv_filename', default="" , type=str,
                    help='csv file name')
    parser.add_argument('--image_key', default="filename" , type=str,
                    help='csv image path columns')
    parser.add_argument('--root_path', default="/home/share/Uni_Eval/Derm7pt/images/", type=str,
                        help='image root path')
    parser.add_argument('--model', default="", type=str,
                        help='image root path')
    parser.add_argument('--percent_data', default=1, type=float)
    parser.add_argument('--batch_size', default=200, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='path to the model checkpoint to load')
    return parser

def main(args):
    model, eval_transform = get_encoder(args.model)
    _ = model.eval()
    model = model.to(device)
    csv_filename = re.sub(r'^.*/', '', args.csv_filename)

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, weights_only=False)['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loading checkpoint from {args.checkpoint}")

    df = pd.read_csv(args.csv_path)
    dataset_train = Derm_Dataset(df=df,
                                root=args.root_path,
                                image_key=args.image_key,
                                train=True,
                                transforms=eval_transform,
                                data_percent=args.percent_data)
    dataset_val = Derm_Dataset(df=df,
                              root=args.root_path,
                              image_key=args.image_key,
                              val=True,
                              transforms=eval_transform)

    dataset_test = Derm_Dataset(df=df,
                               root=args.root_path,
                               image_key=args.image_key,
                               test=True,
                               transforms=eval_transform)

    print('train size:', len(dataset_train), ',val size:', len(dataset_val), ',test size:', len(dataset_test))

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    start = time.time()
    # extract features from the train and test datasets (returns dictionary of embeddings and labels)
    train_features = extract_features_from_dataloader(args, model, train_dataloader)
    val_features = extract_features_from_dataloader(args, model, val_dataloader)
    test_features = extract_features_from_dataloader(args, model, test_dataloader)

    # convert these to torch
    train_feats = torch.Tensor(train_features['embeddings'])
    train_labels = torch.Tensor(train_features['labels']).type(torch.long)
    val_feats = torch.Tensor(val_features['embeddings'])
    val_labels = torch.Tensor(val_features['labels']).type(torch.long)
    test_feats = torch.Tensor(test_features['embeddings'])
    test_labels = torch.Tensor(test_features['labels']).type(torch.long)
    test_filenames = test_features['filenames']
    elapsed = time.time() - start
    print(f'Took {elapsed:.03f} seconds')
    """linear evaluation"""
    from panderm_model.downstream.eval_features.linear_probe import eval_linear_probe
    dataset_name=str(args.csv_path).split('/')[-1].split('.')[0]
    for i in range(1):
        linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
            train_feats=train_feats,
            train_labels=train_labels,
            valid_feats=None,
            valid_labels=val_labels,
            test_feats=test_feats,
            test_labels=test_labels,
            test_filenames=test_filenames,
            max_iter=1000,
            verbose=True, seed=i,
            out_dir=args.output_dir,
            dataset_name=dataset_name
        )

        print_metrics(linprobe_eval_metrics)
        record_metrics_to_csv(linprobe_eval_metrics, dataset_name, csv_filename,args.output_dir)

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)

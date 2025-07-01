#!/usr/bin/env python3
import os
import csv
from prettytable import PrettyTable

# Base directory where the output logs are stored
base_dir = 'linear_probing_logs/'

# List to hold all the summary data
summary_data = []

# Traverse the base directory
for model_dataset in os.listdir(base_dir):
    model_dataset_dir = os.path.join(base_dir, model_dataset)
    if os.path.isdir(model_dataset_dir):
        # Split the directory name to get model and dataset
        model_dataset_split = model_dataset.split('-', 1)
        if len(model_dataset_split) != 2:
            print(f"Invalid model-dataset directory name: {model_dataset}")
            continue
        model, dataset = model_dataset_split

        # Simplify the dataset name by removing the model name and any common prefixes
        # Remove the model name from the dataset if it's included
        dataset = dataset.replace(model, '').lstrip('-').strip()

        # Traverse the checkpoint directories
        for checkpoint_dir in os.listdir(model_dataset_dir):
            checkpoint_path = os.path.join(model_dataset_dir, checkpoint_dir)
            if os.path.isdir(checkpoint_path):
                # Construct the path to the result CSV file
                csv_filename = f"{model}-{dataset}_results.csv"
                csv_filepath = os.path.join(checkpoint_path, csv_filename)
                if os.path.exists(csv_filepath):
                    with open(csv_filepath, 'r') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            # Simplify the checkpoint name
                            # Extract epoch number or any relevant info
                            epoch_info = ''
                            if 'epoch_' in checkpoint_dir:
                                parts = checkpoint_dir.split('epoch_')
                                if len(parts) > 1:
                                    epoch_num = parts[1].split('_')[0]  # Get the number after 'epoch_'
                                    epoch_info = 'Epoch ' + epoch_num
                                else:
                                    epoch_info = checkpoint_dir
                            else:
                                epoch_info = checkpoint_dir  # Use as is if no epoch info

                            data = {
                                'Model': model,
                                'Dataset': dataset,
                                'Checkpoint': epoch_info,
                                'W_F1': row.get('W_F1', ''),
                                'AUROC': row.get('AUROC', ''),
                                'BACC': row.get('BACC', ''),
                                'ACC': row.get('ACC', ''),
                                'AUPR': row.get('AUPR', '')
                            }
                            summary_data.append(data)
                else:
                    print(f"CSV file not found: {csv_filepath}")

# Sort the summary data by Model, Dataset, and Checkpoint
summary_data.sort(key=lambda x: (x['Model'], x['Dataset'], x['Checkpoint']))

# Create a PrettyTable instance
fieldnames = ['Model', 'Dataset', 'Checkpoint', 'W_F1', 'AUROC', 'BACC', 'ACC', 'AUPR']
table = PrettyTable()
table.field_names = fieldnames
table.align = 'l'

# Add rows to the table with formatted numeric values
for data in summary_data:
    # Format numeric values to four decimal places
    for metric in ['W_F1', 'AUROC', 'BACC', 'ACC', 'AUPR']:
        if data[metric]:
            try:
                data[metric] = f"{float(data[metric]):.4f}"
            except ValueError:
                pass  # Keep the original value if it can't be converted to float

    row = [data.get(field, '') for field in fieldnames]
    table.add_row(row)

# Print the table
print(table)

# Optionally, write the table to a text file
output_txt = 'linear_probing_logs/summary_results.txt'
with open(output_txt, 'w') as txtfile:
    txtfile.write(str(table))

print(f"Summary results written to {output_txt}")

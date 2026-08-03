import os
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score


def calculate_thresholds(data: pd.DataFrame) -> float:

    machine_scores = data.query('is_machine==1')['curvature']
    human_scores = data.query('is_machine==0')['curvature']
    # print(len(machine_scores), len(human_scores))
    print(f'------ Human scores: mean {np.mean(human_scores):.4f}, std {np.std(human_scores):.4f} ------')
    print(f'------ Machine scores: mean {np.mean(machine_scores):.4f}, std {np.std(machine_scores):.4f} ------')
    # print(machine_scores.isna().sum(), human_scores.isna().sum())
    labels = [0] * len(human_scores) + [1] * len(machine_scores)
    scores = pd.concat([human_scores, machine_scores])
    print(f'------ AUROC {roc_auc_score(labels, scores):.4f} ------')

    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = np.argmax(tpr - fpr)  # maximises separation
    best_threshold = thresholds[best_idx]
    print(f'------ Suggested threshold: {best_threshold} -------')
    return best_threshold

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Fast-DetectGPT thresholds')
    
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--results-dir', type=str, default='Results Files', required=False)

    args = parser.parse_args()

    if args.file not in os.listdir(args.results_dir):
        raise RuntimeError(f'File {args.file} is not in directory {args.results_dir}')

    df = pd.read_csv(f'{args.results_dir}/{args.file}')
    threshold = calculate_thresholds(df)

    df['prediction'] = [1 if df['curvature'].iloc[i] < threshold else 0 for i in range(len(df))]  # todo: double check this had the intended effect

    y, y_pred = df['is_machine'], df['prediction']

    print()
    print('------ Results with suggested threshold -------')
    print(f'------ Accuracy: {accuracy_score(y, y_pred):.4f} -------')
    print(f'------ Precision: {precision_score(y, y_pred):.4f} ------')
    print(f'------ Recall: {recall_score(y, y_pred):.4f} ------')
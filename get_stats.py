import os

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, mannwhitneyu, friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score

from get_all_data import get_all_result_file_data


all_data = get_all_result_file_data()
generated_directory = 'Generated Datasets'

def get_curvature_stats_by_class() -> None:
    machine = all_data[all_data['is_machine'] == 1]['curvature']
    human = all_data[all_data['is_machine'] == 0]['curvature']

    print('------ Machine Stats ------')
    print(
        f'Mean: {machine.mean()}\n'
        f'StD: {machine.std()}\n'
        f'1stQ {machine.quantile(.25)}\n'
        f'3rdQ {machine.quantile(.75)}'
    )
    print('------ Human Stats ------')
    print(
        f'Mean: {human.mean()}\n'
        f'StD: {human.std()}\n'
        f'1stQ {human.quantile(.25)}\n'
        f'3rdQ {human.quantile(.75)}'
    )
    

def get_curvature_stats_by_dataset() -> None:
    machine = all_data[all_data['is_machine'] == 1]
    human = all_data[all_data['is_machine'] == 0]

    xsum_machine = machine[machine['dataset'] == 'xsum']['curvature']
    wp_machine = machine[machine['dataset'] == 'writingprompts']['curvature']
    wmt_machine = machine[machine['dataset'] == 'wmt16']['curvature']
    squad_machine = machine[machine['dataset'] == 'squad']['curvature']
    pubmedqa_machine = machine[machine['dataset'] == 'pubmedqa']['curvature']

    xsum_human = human[human['dataset'] == 'xsum']['curvature']
    wp_human = human[human['dataset'] == 'writingprompts']['curvature']
    wmt_human = human[human['dataset'] == 'wmt16']['curvature']
    squad_human = human[human['dataset'] == 'squad']['curvature']
    pubmedqa_human = human[human['dataset'] == 'pubmedqa']['curvature']

    print('------ XSum Machine Stats ------')
    print(
        f'Mean: {xsum_machine.mean()}\n'
        f'StD: {xsum_machine.std()}\n'
        f'1stQ {xsum_machine.quantile(.25)}\n'
        f'3rdQ {xsum_machine.quantile(.75)}'
    )
    print('------ Writing Prompts Machine Stats ------')
    print(
        f'Mean: {wp_machine.mean()}\n'
        f'StD: {wp_machine.std()}\n'
        f'1stQ {wp_machine.quantile(.25)}\n'
        f'3rdQ {wp_machine.quantile(.75)}'
    )
    print('------ WMT16 Machine Stats ------')
    print(
        f'Mean: {wmt_machine.mean()}\n'
        f'StD: {wmt_machine.std()}\n'
        f'1stQ {wmt_machine.quantile(.25)}\n'
        f'3rdQ {wmt_machine.quantile(.75)}'
    )
    print('------ SQuAD Machine Stats ------')
    print(
        f'Mean: {squad_machine.mean()}\n'
        f'StD: {squad_machine.std()}\n'
        f'1stQ {squad_machine.quantile(.25)}\n'
        f'3rdQ {squad_machine.quantile(.75)}'
    )
    print('------ PubMedQA Machine Stats ------')
    print(
        f'Mean: {pubmedqa_machine.mean()}\n'
        f'StD: {pubmedqa_machine.std()}\n'
        f'1stQ {pubmedqa_machine.quantile(.25)}\n'
        f'3rdQ {pubmedqa_machine.quantile(.75)}'
    )
    print('------ XSum Human Stats ------')
    print(
        f'Mean: {xsum_human.mean()}\n'
        f'StD: {xsum_human.std()}\n'
        f'1stQ {xsum_human.quantile(.25)}\n'
        f'3rdQ {xsum_human.quantile(.75)}'
    )
    print('------ Writing Prompts Human Stats ------')
    print(
        f'Mean: {wp_human.mean()}\n'
        f'StD: {wp_human.std()}\n'
        f'1stQ {wp_human.quantile(.25)}\n'
        f'3rdQ {wp_human.quantile(.75)}'
    )
    print('------ WMT16 Human Stats ------')
    print(
        f'Mean: {wmt_human.mean()}\n'
        f'StD: {wmt_human.std()}\n'
        f'1stQ {wmt_human.quantile(.25)}\n'
        f'3rdQ {wmt_human.quantile(.75)}'
    )
    print('------ SQuAD Human Stats ------')
    print(
        f'Mean: {squad_human.mean()}\n'
        f'StD: {squad_human.std()}\n'
        f'1stQ {squad_human.quantile(.25)}\n'
        f'3rdQ {squad_human.quantile(.75)}'
    )
    print('------ PubMedQA Human Stats ------')
    print(
        f'Mean: {pubmedqa_human.mean()}\n'
        f'StD: {pubmedqa_human.std()}\n'
        f'1stQ {pubmedqa_human.quantile(.25)}\n'
        f'3rdQ {pubmedqa_human.quantile(.75)}'
    )


def get_curvature_stats_by_model() -> None:
    machine = all_data[all_data['is_machine'] == 1]
    human = all_data[all_data['is_machine'] == 0]

    qwen_machine = machine[machine['sampling_model'] == 'qwen']['curvature']
    bitnet_machine = machine[machine['sampling_model'] == 'bitnet']['curvature']
    stablelm_machine = machine[machine['sampling_model'] == 'stablelm']['curvature']

    qwen_human = human[human['sampling_model'] == 'qwen']['curvature']
    bitnet_human = human[human['sampling_model'] == 'bitnet']['curvature']
    stablelm_human = human[human['sampling_model'] == 'stablelm']['curvature']

    print('------ Qwen Machine Stats ------')
    print(
        f'Mean: {qwen_machine.mean()}\n'
        f'StD: {qwen_machine.std()}\n'
        f'1stQ {qwen_machine.quantile(.25)}\n'
        f'3rdQ {qwen_machine.quantile(.75)}'
    )
    print('------ Bitnet Machine Stats ------')
    print(
        f'Mean: {bitnet_machine.mean()}\n'
        f'StD: {bitnet_machine.std()}\n'
        f'1stQ {bitnet_machine.quantile(.25)}\n'
        f'3rdQ {bitnet_machine.quantile(.75)}'
    )
    print('------ Stable LM Machine Stats ------')
    print(
        f'Mean: {stablelm_machine.mean()}\n'
        f'StD: {stablelm_machine.std()}\n'
        f'1stQ {stablelm_machine.quantile(.25)}\n'
        f'3rdQ {stablelm_machine.quantile(.75)}'
    )

    print('------ Qwen Human Stats ------')
    print(
        f'Mean: {qwen_human.mean()}\n'
        f'StD: {qwen_human.std()}\n'
        f'1stQ {qwen_human.quantile(.25)}\n'
        f'3rdQ {qwen_human.quantile(.75)}'
    )
    print('------ Bitnet Human Stats ------')
    print(
        f'Mean: {bitnet_human.mean()}\n'
        f'StD: {bitnet_human.std()}\n'
        f'1stQ {bitnet_human.quantile(.25)}\n'
        f'3rdQ {bitnet_human.quantile(.75)}'
    )
    print('------ Stable LM Human Stats ------')
    print(
        f'Mean: {stablelm_human.mean()}\n'
        f'StD: {stablelm_human.std()}\n'
        f'1stQ {stablelm_human.quantile(.25)}\n'
        f'3rdQ {stablelm_human.quantile(.75)}'
    )



def do_anova_test() -> tuple[float, float]:

    machine = all_data[all_data['is_machine'] == 1]
    human = all_data[all_data['is_machine'] == 0]['curvature']

    qwen = machine[machine['sampling_model'] == 'qwen']['curvature']
    bitnet = machine[machine['sampling_model'] == 'bitnet']['curvature']
    stablelm = machine[machine['sampling_model'] == 'stablelm']['curvature']

    # plt.boxplot([human.to_list(), qwen.to_list(), bitnet.to_list(), stablelm.to_list()], tick_labels=['Human', 'Qwen', 'Bitnet', 'Stable LM'])
    # plt.show()

    return f_oneway(human, bitnet, qwen, stablelm, nan_policy='omit', equal_var=False)

def do_anova_test_on_iqr() -> tuple[float, float]:

    machine = all_data[all_data['is_machine'] == 1]
    human = all_data[all_data['is_machine'] == 0]

    machine = machine[
        (machine['curvature'] >= machine['curvature'].quantile(.25)) &
        (machine['curvature'] <= machine['curvature'].quantile(.75))
    ]

    human = human[
        (human['curvature'] >= human['curvature'].quantile(.25)) &
        (human['curvature'] <= human['curvature'].quantile(.75))
    ]['curvature']

    qwen = machine[machine['sampling_model'] == 'qwen']['curvature']
    bitnet = machine[machine['sampling_model'] == 'bitnet']['curvature']
    stablelm = machine[machine['sampling_model'] == 'stablelm']['curvature']

    return f_oneway(human, bitnet, qwen, stablelm, nan_policy='omit', equal_var=False)


def do_t_test() -> zip:

    df = all_data.dropna(subset=['curvature'])
    machine = df[df['is_machine'] == 1]
    # human = df[df['is_machine'] == 0]['curvature']

    qwen = machine[machine['sampling_model'] == 'qwen']['curvature']
    bitnet = machine[machine['sampling_model'] == 'bitnet']['curvature']
    stablelm = machine[machine['sampling_model'] == 'stablelm']['curvature']

    qwen_human = df[
        (df['is_machine'] == 0) &
        (df['sampling_model'] == 'qwen')
    ]['curvature']

    bitnet_human = df[
        (df['is_machine'] == 0) &
        (df['sampling_model'] == 'bitnet')
    ]['curvature']

    stablelm_human = df[
        (df['is_machine'] == 0) &
        (df['sampling_model'] == 'stablelm')
    ]['curvature']

    df_pairs = [
        (qwen_human, qwen),
        (bitnet_human, bitnet),
        (stablelm_human, stablelm),
        # (qwen, bitnet),
        # (qwen, stablelm),
        # (bitnet, stablelm),
    ]


    t_results = []
    for dfs in df_pairs:
        a, b = dfs
        result = mannwhitneyu(a, b, nan_policy='omit')
        t_results.append(result)

    rejected, corrected_p, _, _ = multipletests(
        [p for _, p in t_results],
        alpha=.05,
        method='bonferroni'
    )

    cohens_d = []
    def calculate_cohens_d(a, b):
        return (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2)

    for a, b in df_pairs:
        cohens_d.append(calculate_cohens_d(a, b))

    t_statistics = []
    p_values = []
    for t, p in t_results:
        t_statistics.append(t)
        p_values.append(p)

    return zip(t_statistics, p_values, rejected, corrected_p, cohens_d)


def do_friedman_test() -> None:
    df = all_data

    qwen_machine = df[
        (df['is_machine'] == 1) &
        (df['sampling_model'] == 'qwen')
    ]['curvature']

    bitnet_machine = df[
        (df['is_machine'] == 1) &
        (df['sampling_model'] == 'bitnet')
    ]['curvature']

    stablelm_machine = df[
        (df['is_machine'] == 1) &
        (df['sampling_model'] == 'stablelm')
    ]['curvature']

    qwen_human = df[
        (df['is_machine'] == 0) &
        (df['sampling_model'] == 'qwen')
    ]['curvature']

    bitnet_human = df[
        (df['is_machine'] == 0) &
        (df['sampling_model'] == 'bitnet')
    ]['curvature']

    stablelm_human = df[
        (df['is_machine'] == 0) &
        (df['sampling_model'] == 'stablelm')
    ]['curvature']

    print('Friedman Machine Curvature Comparison')
    print(friedmanchisquare(qwen_machine, bitnet_machine, stablelm_machine, nan_policy='omit'))
    print('Friedman Human Curvature Comparison')
    print(friedmanchisquare(qwen_human, bitnet_human, stablelm_human, nan_policy='omit'))


def do_wilcoxon_test() -> None:
    df = all_data

    qwen_machine = df[
        (df['is_machine'] == 1) &
        (df['sampling_model'] == 'qwen')
    ]['curvature']

    bitnet_machine = df[
        (df['is_machine'] == 1) &
        (df['sampling_model'] == 'bitnet')
    ]['curvature']

    stablelm_machine = df[
        (df['is_machine'] == 1) &
        (df['sampling_model'] == 'stablelm')
    ]['curvature']

    qwen_human = df[
        (df['is_machine'] == 0) &
        (df['sampling_model'] == 'qwen')
    ]['curvature']

    bitnet_human = df[
        (df['is_machine'] == 0) &
        (df['sampling_model'] == 'bitnet')
    ]['curvature']

    stablelm_human = df[
        (df['is_machine'] == 0) &
        (df['sampling_model'] == 'stablelm')
    ]['curvature']

    test_pairs = [
        (qwen_machine, bitnet_machine),
        (qwen_machine, stablelm_machine),
        (stablelm_machine, bitnet_machine),
        (qwen_human, bitnet_human),
        (qwen_human, stablelm_human),
        (stablelm_human, bitnet_human)
    ]

    titles = [
        ('Machine', 'Qwen vs. Bitnet'),
        ('Machine', 'Qwen vs. Stable LM'),
        ('Machine', 'Stable LM vs. Bitnet'),
        ('Human', 'Qwen vs. Bitnet'),
        ('Human', 'Qwen vs. Stable LM'),
        ('Human', 'Stable LM vs. Bitnet')
    ]

    stats = []
    p_values = []
    for model_a, model_b in test_pairs:
        stat, p = wilcoxon(model_a, model_b, nan_policy='omit')
        stats.append(stat)
        p_values.append(p)


    reject, p_adjusted, _, _ = multipletests(p_values, alpha=.05, method='bonferroni')
    for t, p, r, corrected, (title_a, title_b) in zip(stats, p_values, reject, p_adjusted, titles):
        print(f'Wilcoxon {title_a} Curvature Comparison {title_b}')
        print(f'statistic: {t}')
        print(f'p-value: {p}')
        print(f'Reject null hypothesis: {r}')
        print(f'Corrected p-values: {corrected}')
        print()
    
def get_average_aurocs() -> None:
    df = all_data.dropna(subset=['curvature'])

    def group_auroc_calculation(group):
        return roc_auc_score(group['is_human'], group['curvature'])

    df = df.groupby(['sampling_model', 'dataset', 'test']).apply(group_auroc_calculation).reset_index(name='auroc')
    print('AUROC by Model')
    print(f'Mean:\n{df.groupby(['sampling_model', 'test'])['auroc'].mean()}')
    print('\nAUROC by Dataset')
    print(f'Mean:\n{df.groupby(['dataset', 'test'])['auroc'].mean()}')


def get_average_document_length_by_dataset(human: bool) -> None:

    xsum = []
    wp = []
    wmt = []
    squad = []
    pubmedqa = []

    for file in os.listdir(generated_directory):
        df = pd.read_parquet(f'{generated_directory}/{file}')

        if human:
            df = df[df['is_machine'] == 0]['document']
        else:
            df = df[df['is_machine'] == 1]['document']
        lens = df.apply(lambda x: len(x)).to_list()

        if 'xsum' in file:
            xsum.extend(lens)
        elif 'writingprompts' in file:
            wp.extend(lens)
        elif 'wmt16' in file:
            wmt.extend(lens)
        elif 'squad' in file:
            squad.extend(lens)
        else:
            pubmedqa.extend(lens)


    print(f'{'Human' if human else 'Machine'} stats for documents by dataset')
    print(f'XSum Length Mean: {np.average(xsum)}')
    print(f'XSum Length StD: {np.std(xsum)}')
    print(f'XSum Length Median: {np.median(xsum)}')
    print(f'Writing Prompts Length Mean: {np.average(wp)}')
    print(f'Writing Prompts Length StD: {np.std(wp)}')
    print(f'Writing Prompts Length Median: {np.median(wp)}')
    print(f'WMT16 DE–EN Length Mean: {np.average(wmt)}')
    print(f'WMT16 DE–EN Length StD: {np.std(wmt)}')
    print(f'WMT16 DE–EN Median: {np.median(wmt)}')
    print(f'SQuAD Length Mean: {np.average(squad)}')
    print(f'SQuAD Length StD: {np.std(squad)}')
    print(f'SQuAD Length Median: {np.median(squad)}')
    print(f'PubMedQA Length Mean: {np.average(pubmedqa)}')
    print(f'PubMedQA Length StD: {np.std(pubmedqa)}')
    print(f'PubMedQA Length Median: {np.median(pubmedqa)}')


# calculate classification metrics
def calculate_thresholds(human_scores: pd.Series, machine_scores: pd.Series) -> float:

    print(f'------ Human scores: mean {np.mean(human_scores):.4f}, std {np.std(human_scores):.4f} ------')
    print(f'------ Machine scores: mean {np.mean(machine_scores):.4f}, std {np.std(machine_scores):.4f} ------')

    labels = [1] * len(human_scores) + [0] * len(machine_scores)
    scores = pd.concat([human_scores, machine_scores])
    print(f'------ AUROC {roc_auc_score(labels, scores):.4f} ------')

    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = np.argmax(tpr - fpr)  # maximizes separation
    best_threshold = thresholds[best_idx]
    print(f'------ Suggested threshold: {best_threshold} -------')
    return best_threshold


def calculate_metrics(y_true, y_pred, y_score):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, y_score),
    }


def get_grouped_classification_metrics():
    train_results = []
    test_results = []
    df = all_data.dropna(subset=['curvature'])
    for (model, dataset), group in df.groupby(['sampling_model', 'dataset']):
        train = group[group['test'] == 0]
        test = group[group['test'] == 1]

        train_human = train[train['is_human'] == 1]['curvature']
        train_machine = train[train['is_human'] == 0]['curvature']

        threshold = calculate_thresholds(train_human, train_machine)
        print(f'Threshold: {threshold}\nModel: {model}\nDataset: {dataset}')
        
        train_preds = (train['curvature'] >= threshold).astype(int)
        train_metrics = calculate_metrics(train['is_human'], train_preds, train['curvature'])
        train_results.append({
            'model': model,
            'dataset': dataset,
            'threshold': threshold,
            **train_metrics
        })
        
        test_preds = (test['curvature'] >= threshold).astype(int)
        test_metrics = calculate_metrics(test['is_human'], test_preds, test['curvature'])
        test_results.append({
            'model': model,
            'dataset': dataset,
            'threshold': threshold,
            **test_metrics
        })
    
    train_results_df = pd.DataFrame(train_results)
    test_results_df = pd.DataFrame(test_results)

    train_by_model = train_results_df.groupby('model')[['accuracy', 'precision', 'recall', 'f1', 'auroc']].mean().round(3)
    train_by_dataset = train_results_df.groupby('dataset')[['accuracy', 'precision', 'recall', 'f1', 'auroc']].mean().round(3)

    test_by_model = test_results_df.groupby('model')[['accuracy', 'precision', 'recall', 'f1', 'auroc']].mean().round(3)
    test_by_dataset = test_results_df.groupby('dataset')[['accuracy', 'precision', 'recall', 'f1', 'auroc']].mean().round(3)

    print()
    print(train_by_model.to_string())
    print(train_by_dataset.to_string())
    print()
    print(test_by_model.to_string())
    print(test_by_dataset.to_string())
    print('------------------')
    print(train_results_df.groupby('model')['threshold'].mean())
    print(test_results_df.groupby('dataset')['threshold'].mean())

get_curvature_stats_by_class()
print()
get_curvature_stats_by_dataset()
print()
get_curvature_stats_by_model()
print()
print(do_anova_test())
print(do_anova_test_on_iqr())
print()
for t, p, reject, corrected, cohens in do_t_test():
    print(f't-statistic: {t}')
    print(f'p-value: {p}')
    print(f'Reject null hypothesis: {reject}')
    print(f'Corrected p-values: {corrected}')
    print(f'Cohen\'s d: {cohens}')
    print()

do_friedman_test()
print()
do_wilcoxon_test()
print()
get_average_aurocs()
print()
get_average_document_length_by_dataset(human=False)  # machine
print()
get_average_document_length_by_dataset(human=True)
print()
get_grouped_classification_metrics()
print()
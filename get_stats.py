from get_all_data import get_all_result_file_data
from scipy.stats import f_oneway, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score
import numpy as np


all_data = get_all_result_file_data()
print(all_data.columns)

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
    human = df[df['is_machine'] == 0]['curvature']

    qwen = machine[machine['sampling_model'] == 'qwen']['curvature']
    bitnet = machine[machine['sampling_model'] == 'bitnet']['curvature']
    stablelm = machine[machine['sampling_model'] == 'stablelm']['curvature']

    df_pairs = [
        (human, qwen),
        (human, bitnet),
        (human, stablelm),
        (qwen, bitnet),
        (qwen, stablelm),
        (bitnet, stablelm),
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


def get_average_aurocs() -> None:
    df = all_data.dropna(subset=['curvature'])

    def group_auroc_calculation(group):
        return 1 - roc_auc_score(group['is_machine'], group['curvature'])

    df = df.groupby(['sampling_model', 'dataset', 'test']).apply(group_auroc_calculation).reset_index(name='auroc')
    print('AUROC by Model')
    print(df.groupby(['sampling_model', 'test'])['auroc'].mean())
    print('\nAUROC by Dataset')
    print(df.groupby(['dataset', 'test'])['auroc'].mean())


get_curvature_stats_by_class()
print()
get_curvature_stats_by_dataset()
print()
print(do_anova_test())
print(do_anova_test_on_iqr())

for t, p, reject, corrected, cohens in do_t_test():
    print(f't-statistic: {t}')
    print(f'p-value: {p}')
    print(f'Reject null hypothesis: {reject}')
    print(f'Corrected p-values: {corrected}')
    print(f'Cohen\'s d: {cohens}')
    print()

get_average_aurocs()
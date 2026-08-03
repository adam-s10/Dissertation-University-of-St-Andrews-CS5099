from get_all_data import get_all_result_file_data

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc

results_dir = 'Results Files'


def plot_is_machine_time_stats() -> None:
    no_nans = all_data.dropna(subset=['total_run_time_seconds'])

    no_nans['total_run_time_seconds'] = pd.to_timedelta(no_nans['total_run_time_seconds'])
    no_nans['total_run_time_seconds'] = no_nans['total_run_time_seconds'].dt.microseconds

    machine = no_nans[no_nans['is_machine'] == 1]
    human = no_nans[no_nans['is_machine'] == 0]

    machine_runtime = machine['total_run_time_seconds']
    human_runtime = human['total_run_time_seconds']

    _, (ax_machine, ax_human) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

    for df, ax, title in zip([machine_runtime, human_runtime], [ax_machine, ax_human], ['Machine', 'Human']):
        df.hist(ax=ax, bins=20, edgecolor='black')
        ax.set_title(f'Runtime for {title} Samples (μs)')

    plt.tight_layout()
    plt.show()


def plot_curvature_stats() -> None:
    machine = all_data[all_data['is_machine'] == 1]
    human = all_data[all_data['is_machine'] == 0]

    machine_curvature = machine['curvature']
    human_curvature = human['curvature']

    fig, (ax_machine, ax_human) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for df, ax, color in zip(
            [machine_curvature, human_curvature],
            [ax_machine, ax_human],
            ['#ff7f0e', '#1f77b4']  # default hex values of matplot when plotting two histograms on the same axis
    ):
        df.hist(ax=ax, color= color, edgecolor='white', alpha=.4)


    fig.supxlabel(t='Conditional Probability Curvature', fontsize='large', fontweight='bold')
    fig.supylabel(t='Frequency', fontsize='large', fontweight='bold')
    fig.legend(['Machine', 'Human'], loc='lower left', ncols=2)
    plt.tight_layout()
    plt.savefig(fname='human_and_machine_curvatures')
    plt.show()


def plot_curvature_stats_by_dataset() -> None:
    machine = all_data[all_data['is_machine'] == 1]
    human = all_data[all_data['is_machine'] == 0]

    xsum_machine_curvature = machine[machine['dataset'] == 'xsum']['curvature']
    wp_machine_curvature = machine[machine['dataset'] == 'writingprompts']['curvature']
    wmt_machine_curvature = machine[machine['dataset'] == 'wmt16']['curvature']
    squad_machine_curvatures = machine[machine['dataset'] == 'squad']['curvature']
    pubmedqa_machine_curvatures = machine[machine['dataset'] == 'pubmedqa']['curvature']

    xsum_human_curvature = human[human['dataset'] == 'xsum']['curvature']
    wp_human_curvature = human[human['dataset'] == 'writingprompts']['curvature']
    wmt_human_curvature = human[human['dataset'] == 'wmt16']['curvature']
    squad_human_curvatures = human[human['dataset'] == 'squad']['curvature']
    pubmedqa_human_curvatures = human[human['dataset'] == 'pubmedqa']['curvature']

    fig, (ax_xsum, ax_wp, ax_wmt, ax_squad, ax_pubmedqa) = plt.subplots(1, 5, figsize=(14, 6), )

    plot_zip = zip(
        [
            (xsum_machine_curvature, xsum_human_curvature),
            (wp_machine_curvature, wp_human_curvature),
            (wmt_machine_curvature, wmt_human_curvature),
            (squad_machine_curvatures, squad_human_curvatures),
            (pubmedqa_machine_curvatures, pubmedqa_human_curvatures)
        ],
        [
            ax_xsum,
            ax_wp,
            ax_wmt,
            ax_squad,
            ax_pubmedqa
        ],
        [
            'XSum',
            'Writing Prompts',
            'WMT16 DE-EN',
            'SQuAD',
            'PubMedQA'
        ],
        [
            ('orange', 'blue'),
            ('orange', 'blue'),
            ('orange', 'blue'),
            ('orange', 'blue'),
            ('orange', 'blue')
        ]
    )
    for df, ax, title, color in plot_zip:

        machine_df, human_df = df
        machine_color, human_color = None, None
        human_df.hist(ax=ax, color=human_color, alpha=.4, edgecolor="white")
        machine_df.hist(ax=ax, color=machine_color, alpha=.4, edgecolor='white')
        ax.set_title(title, fontsize='medium', style='italic')

    fig.supxlabel(t='Conditional Probability Curvature', fontsize='large', fontweight='bold')
    fig.supylabel(t='Frequency', fontsize='large', fontweight='bold')
    fig.legend(['Human', 'Machine'], loc='lower left', ncols=2)
    plt.tight_layout()
    # plt.savefig(fname='human_and_machine_by_dataset_curvatures_different_scales')
    plt.show()
    # todo: change graphs scale to be one standard deviation

def plot_curvature_iqr() -> None:
    machine = all_data[all_data["is_machine"] == 1]
    human = all_data[all_data["is_machine"] == 0]

    machine_curvature = machine["curvature"]
    human_curvature = human["curvature"]

    machine_curvature_iqr = machine_curvature[
        (machine_curvature >= machine_curvature.quantile(0.25))
        & (machine_curvature <= machine_curvature.quantile(0.75))
    ]

    human_curvature_iqr = human_curvature[
        (human_curvature >= human_curvature.quantile(0.25))
        & (human_curvature <= human_curvature.quantile(0.75))
    ]

    _, (ax_machine, ax_human) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for df, ax, title in zip(
        [machine_curvature_iqr, human_curvature_iqr],
        [ax_machine, ax_human],
        ["Machine", "Human"],
    ):
        df.hist(ax=ax, bins=20, edgecolor="black")
        ax.set_title(f"Curvature for {title} (IQR)")

    plt.tight_layout()
    plt.show()


def plot_curvature_by_model() -> None:
    machine = all_data[all_data['is_machine'] == 1]
    human = all_data[all_data['is_machine'] == 0]

    qwen_machine = machine[machine['sampling_model'] == 'qwen']['curvature']
    bitnet_machine = machine[machine['sampling_model'] == 'bitnet']['curvature']
    stablelm_machine = machine[machine['sampling_model'] == 'stablelm']['curvature']

    qwen_human = human[human['sampling_model'] == 'qwen']['curvature']
    bitnet_human = human[human['sampling_model'] == 'bitnet']['curvature']
    stablelm_human = human[human['sampling_model'] == 'stablelm']['curvature']

    fig, (
        (top_left, top_right),
        (middle_left, middle_right),
        (bottom_left, bottom_right)
    ) = plt.subplots(3, 2, figsize=(14, 6), sharey=True)

    plot_zip = zip(
        [qwen_machine, qwen_human, bitnet_machine, bitnet_human, stablelm_machine, stablelm_human],
        [top_left, top_right, middle_left, middle_right, bottom_left, bottom_right],
        ['Machine Samples', 'Human Samples', None, None, None, None],
        ['red', 'red', 'green', 'green', 'blue', 'blue']
    )

    for df, ax, title, color in plot_zip:
        df.hist(ax=ax, bins=20, color=color, edgecolor='white', alpha=.4)
        ax.set_title(f'Distribution of Curvatures using {title} Samples')
        ax.set_title(title, fontsize='medium', style='italic')


    fig.supxlabel(t='Conditional Probability Curvature', fontsize='large', fontweight='bold')
    fig.supylabel(t='Frequency', fontsize='large', fontweight='bold')

    handles = [
        Patch(color='red', edgecolor='white',   alpha=.4, label='Qwen'),
        Patch(color='green', edgecolor='white', alpha=.4, label='Bitnet'),
        Patch(color='blue', edgecolor='white',  alpha=.4, label='Stable LM'),
    ]
    fig.legend(handles=handles, loc='lower left', ncols=3)
    plt.tight_layout()
    plt.savefig('human_and_machine_curvatures_by_model')
    plt.show()


def plot_auroc(y_true: list[tuple[int]], z_scores: list[tuple[float]], dataset: str) -> None:
    _, axis = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    models = ["Qwen", "Bitnet", "Stable LM"]
    colors = ["red", "green", "blue"]

    auc_zip = zip(y_true, z_scores, axis, ["Train", "Test"])
    for true, score, ax, title in auc_zip:
        for model_true, model_score, color, model in zip(
            true, score, colors, models
        ):
            # for model_true, model_score, color in true, score, color:
            fpr, tpr, _ = roc_curve(model_true, model_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, color=color, label=f"{model} (AUROC = {1 - roc_auc:.3f})"
            )
            ax.legend()
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title(f"ROC Curves for {dataset} {title}")
    plt.tight_layout()
    plt.show()


all_data = get_all_result_file_data()
all_data['total_run_time_seconds'] = pd.to_timedelta(all_data['total_run_time_seconds'])
all_data['total_run_time_seconds'] = all_data['total_run_time_seconds'].dt.microseconds
plot_is_machine_time_stats()
plot_curvature_stats()
plot_curvature_iqr()
plot_curvature_stats_by_dataset()
plot_curvature_by_model()
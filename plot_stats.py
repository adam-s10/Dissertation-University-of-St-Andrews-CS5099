from get_all_data import get_all_result_file_data

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc

results_dir = 'Results Files'

MODELS = ['qwen', 'bitnet', 'stablelm']
COLORS = ['red', 'green', 'blue']
DATASET_ORDER = ['xsum', 'writingprompts', 'wmt16', 'squad', 'pubmedqa']
DATASET_TITLES = ['Xsum', 'Writing Prompts', 'WMT16 DE-EN', 'SQuAD', 'PubMedQA']

# Global control of figures for figure generation consistency

# A4 default margin size
FIG_WIDTH = 6.3
FONT_SIZE = 9

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
FIG_HEIGHT = 2.5
DEFAULT_HEX_COLORS = ['#ff7f0e', '#1f77b4']

def plot_curvature_stats() -> None:
    """
    Saves plot of all curvature by class returned by all three detection models.
    *Figure 6*
    :return: None
    """
    machine = all_data[all_data['is_machine'] == 1]
    human = all_data[all_data['is_machine'] == 0]

    machine_curvature = machine['curvature']
    human_curvature = human['curvature']

    fig, (ax_machine, ax_human) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True)

    for df, ax, color in zip(
            [machine_curvature, human_curvature],
            [ax_machine, ax_human],
            DEFAULT_HEX_COLORS
    ):
        df.hist(ax=ax, color= color, edgecolor='white', alpha=.4)


    fig.supxlabel(t='Conditional Probability Curvature', fontweight='bold')
    fig.supylabel(t='Frequency', fontweight='bold')
    fig.legend(['Machine', 'Human'], loc='lower left', ncols=1, frameon=False)
    plt.tight_layout()
    plt.savefig(fname='human_and_machine_curvatures')


def plot_curvature_by_model() -> None:
    """
    Saves plot of all curvature split by class and further split by the three detection models.
    *Figure 7*
    :return: None
    """
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
    ) = plt.subplots(3, 2, figsize=(FIG_WIDTH, FIG_HEIGHT * 3), sharey=True)

    plot_zip = zip(
        [qwen_machine, qwen_human, bitnet_machine, bitnet_human, stablelm_machine, stablelm_human],
        [top_left, top_right, middle_left, middle_right, bottom_left, bottom_right],
        ['Machine Samples', 'Human Samples', None, None, None, None],
        ['red', 'red', 'green', 'green', 'blue', 'blue']
    )

    for df, ax, title, color in plot_zip:
        df.hist(ax=ax, bins=20, color=color, edgecolor='white', alpha=.4)
        ax.set_title(f'Distribution of Curvatures using {title} Samples')
        ax.set_title(title, style='italic')

    handles = [
        Patch(color='red', edgecolor='white',   alpha=.4, label='Qwen'),
        Patch(color='green', edgecolor='white', alpha=.4, label='Bitnet'),
        Patch(color='blue', edgecolor='white',  alpha=.4, label='Stable LM'),
    ]
    fig.legend(
        handles=handles,
        fontsize=FONT_SIZE,
        ncols=3,
        frameon=False,
        loc='lower left'
    )

    fig.supxlabel(t='Conditional Probability Curvature', fontsize=FONT_SIZE, fontweight='bold', y=0.03)
    fig.supylabel(t='Frequency', fontsize=FONT_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname='human_and_machine_curvature_by_model')

def plot_squad_vs_wmt_curvatures() -> None:
    """
    Saves plot of all machine and human curvatures of all three detection models for WMT16 DE-EN and SQuAD datasets.
    *Figure 8*
    :return: None
    """
    machine = all_data[all_data['is_human'] == 0]
    human = all_data[all_data['is_human'] == 1]

    wmt_machine_curvature = machine[machine['dataset'] == 'wmt16']['curvature']
    squad_machine_curvatures = machine[machine['dataset'] == 'squad']['curvature']
    wmt_human_curvature = human[human['dataset'] == 'wmt16']['curvature']
    squad_human_curvatures = human[human['dataset'] == 'squad']['curvature']

    fig, (left, right) = plt.subplots(nrows=1, ncols=2, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True)
    plot_zip = zip(
        [
            (wmt_machine_curvature, wmt_human_curvature),
            (squad_machine_curvatures, squad_human_curvatures)
        ],
        [left, right],
        ['WMT16 DE-EN', 'SQuAD']
    )

    for (machine, human), ax, title in plot_zip:
        machine.hist(ax=ax, color=DEFAULT_HEX_COLORS[0], alpha=.4, edgecolor='white')
        human.hist(ax=ax, color=DEFAULT_HEX_COLORS[1], alpha=.4, edgecolor='white')
        ax.set_title(title, style='italic')

    fig.supylabel(t='Frequency', fontsize=FONT_SIZE, fontweight='bold')
    fig.supxlabel(t='Conditional Probability Curvature', fontsize=FONT_SIZE, fontweight='bold')
    fig.legend(['Machine', 'Human'], loc='lower left', fontsize=FONT_SIZE, ncols=1, frameon=False)
    plt.tight_layout()
    plt.savefig(fname='wmt_vs_squad')


def plot_auroc(generation_model: str) -> None:
    """
    Saves plot ROC curves with respective AUROC scores of all datasets generated by the passed model for all three of
    the detection models.
    *Section C*
    :param generation_model: generation model to plot figure for
    :return: None
    """

    df = all_data[all_data['generation_model'] == generation_model]
    df = df.dropna(subset=['curvature'])  # sklearn AUROC will throw errors if nans exist
    title_preamble = 'ROC Curves for'
    fig, axes = plt.subplots(5, 2, figsize=(FIG_WIDTH, FIG_HEIGHT * 5), sharey=True, sharex=True)

    for dataset, title, ax in zip(DATASET_ORDER, DATASET_TITLES, axes):
        dataset_df = df[df['dataset'] == dataset]
        print(dataset_df.shape)
        for dataset_type, col_val, a, in zip(['Train', 'Test'], [0, 1], ax):
            for model, color in zip(MODELS, COLORS):
                plot_df = dataset_df[
                    (dataset_df['sampling_model'] == model) &
                    (dataset_df['test'] == col_val)
                ]
                model_true = plot_df['is_human']
                model_score = plot_df['curvature']
                fpr, tpr, _ = roc_curve(model_true, model_score)
                roc_auc = auc(fpr, tpr)
                a.plot(
                    fpr, tpr, color=color, label=f'AUROC: {roc_auc:.3f}'
                )
                a.legend(loc='lower right', frameon=False)
                a.set_title(f'{title_preamble} {title} {dataset_type}', style='italic')

    handles = [
        Patch(color='red', edgecolor='white', label='Qwen'),
        Patch(color='green', edgecolor='white', label='Bitnet'),
        Patch(color='blue', edgecolor='white', label='Stable LM'),
    ]

    fig.legend(
        handles=handles,
        fontsize=FONT_SIZE,
        ncols=3,
        frameon=False,
        loc='lower left'
    )
    fig.supxlabel('False Positive Rate (FPR)', fontsize=FONT_SIZE, fontweight='bold', y=0.02)
    fig.supylabel('True Positive Rate (TPR)', fontsize=FONT_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname=f'{generation_model}_roc_curves')

all_data = get_all_result_file_data()

plot_curvature_stats()

plot_squad_vs_wmt_curvatures()

plot_curvature_by_model()

plot_auroc('qwen_ft')
plot_auroc('llama')
plot_auroc('google')
plot_auroc('rocket')
plot_auroc('nvidia_4b')
plot_auroc('nvidia_7b')

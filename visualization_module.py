import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

DPI = 300


def generate_cell_visualization(save_path):

    np.random.seed(42)

    n_healthy = 300
    healthy_x = np.random.normal(2.0, 0.6, n_healthy)
    healthy_y = np.random.normal(2.0, 0.6, n_healthy)

    n_malign = 120
    mal_x = np.concatenate([
        np.random.normal(4.0, 0.4, n_malign // 2),
        np.random.normal(1.0, 0.3, n_malign // 2)
    ])
    mal_y = np.concatenate([
        np.random.normal(4.0, 0.4, n_malign // 2),
        np.random.normal(1.0, 0.3, n_malign // 2)
    ])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)

    ax.scatter(healthy_x, healthy_y, s=18, alpha=0.85, label="Healthy")
    ax.scatter(mal_x, mal_y, s=28, alpha=0.95, label="Leukemic")

    ax.set_title("Microscopic Cell Distribution (Placeholder)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)

__author__ = "Hemanth Sunkara"
__project__ = "Hybrid ACO-GWO Leukemia Framework"
project_name = "HemanthSunkara-ProjectG"

def plot_gwo_hierarchy(wolves, scores, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    wolves = np.asarray(wolves)
    scores = np.asarray(scores)

    if wolves.ndim != 2 or wolves.shape[1] < 2:
        wolves = np.random.rand(10, 2)
        scores = np.random.rand(wolves.shape[0])

    idx = scores.argsort()[::-1]
    alpha_idx, beta_idx, delta_idx = idx[:3]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)

    ax.scatter(wolves[:, 0], wolves[:, 1], s=40, alpha=0.7, label="Omega")
    ax.scatter(wolves[alpha_idx, 0], wolves[alpha_idx, 1], s=120, label="Alpha")
    ax.scatter(wolves[beta_idx, 0], wolves[beta_idx, 1], s=100, label="Beta")
    ax.scatter(wolves[delta_idx, 0], wolves[delta_idx, 1], s=100, label="Delta")

    for i in range(min(8, len(wolves))):
        ax.annotate(
            '',
            xy=wolves[i],
            xytext=wolves[alpha_idx],
            arrowprops=dict(arrowstyle='->', lw=0.8, alpha=0.6)
        )

    ax.set_title("Grey Wolf Optimizer - Hierarchy")
    ax.set_xlabel("Position dim 1")
    ax.set_ylabel("Position dim 2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)


def plot_feature_reduction(original_count, selected_count, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = ["Original Features", "Selected Features"]
    values = [original_count, selected_count]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)

    bars = ax.bar(labels, values, alpha=0.95)

    ax.set_ylabel("Number of Features")
    ax.set_title("Feature Reduction via ACO")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f'{int(h)}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)

def plot_metrics(metrics, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    keys = ["Accuracy", "Sensitivity", "Specificity", "ROC-AUC"]
    vals = [metrics.get(k, 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=DPI)

    bars = ax.bar(keys, vals, alpha=0.95)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Metrics")
    ax.grid(axis='y', alpha=0.12)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)

def draw_pipeline_flowchart(save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    G = nx.DiGraph()

    steps = [
        "Input",
        "Preprocessing",
        "ACO",
        "GWO",
        "Classifier",
        "Evaluation"
    ]

    for i, s in enumerate(steps):
        G.add_node(i, label=s)

    for i in range(len(steps) - 1):
        G.add_edge(i, i + 1)

    pos = {
        0: (0, 0),
        1: (1.6, 0),
        2: (3.2, 0),
        3: (4.8, 0),
        4: (6.4, 0),
        5: (8.0, 0)
    }

    fig, ax = plt.subplots(figsize=(12, 2.5), dpi=DPI)

    nx.draw_networkx_nodes(G, pos, node_size=3600)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20)

    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels)

    ax.set_axis_off()
    ax.set_title("Hybrid ACO-GWO-SVM Workflow")

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)

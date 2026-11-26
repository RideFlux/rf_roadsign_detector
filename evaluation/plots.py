"""Shared confusion-matrix image export for classifier and pipeline evaluation."""

from pathlib import Path

import numpy as np


def save_confusion_matrix(matrix, names, output_path, title='Confusion Matrix'):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(9, 8))
    plot = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(plot, ax=ax)
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=names, yticklabels=names, xlabel='Predicted class',
           ylabel='True class', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

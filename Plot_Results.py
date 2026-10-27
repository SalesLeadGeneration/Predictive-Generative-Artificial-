import numpy as np
from itertools import cycle
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.metrics import roc_curve, confusion_matrix
import os


def ROC_curve():
    lw = 2
    cls = ['Attention guidedDNN', 'CNN', 'Resnet', 'VGG19', 'GDD-GDAN']
    # Create a figure and axis
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Roc Curve')
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')
    Actual = np.reshape(Actual, [-1, 1])
    per = round(Actual.shape[0] * 0.80)
    Actual = Actual[per:, :]
    colors = cycle(["#fe2f4a", "#8b88f8", "#fc824a", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def Plot_Confusion():
    Classes = ['Normal', 'Early \nStage', 'Early Severe\n Stage', 'Mid \nStage', 'Severe\n Stage']
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    cm = confusion_matrix(np.asarray(Actual).argmax(axis=1), np.asarray(Predict).argmax(axis=1))
    # cm = confusion_matrix(np.asarray(Actual), np.asarray(Predict))
    rot = 0
    fontsizes = 12
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap='Blues', values_format='d', text_kw={'fontsize': 12})
    ax.set_xlabel('Predicted labels', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual labels', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
    ax.set_yticklabels(Classes, fontsize=fontsizes)
    plt.tight_layout()
    path = "./Results/Confusion_matrix.png"
    plt.savefig(path)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def Plot_table():
    Eval_act = np.load('Eval_all_batch.npy', allow_pickle=True)
    Graph_Term = np.array([0]).astype(int)
    Networks = ['Batch Size', 'Attention guidedDNN', 'CNN', 'Resnet', 'VGG19', 'GDD-GDAN']
    variation = ['4', '8', '16', '32', '48']
    value_act = Eval_act[:, :, 4:]
    Table = PrettyTable()
    Table.add_column(Networks[0], variation)
    for j in range(len(Networks) - 1):
        if j == 4:
            val = value_act
        else:
            val = value_act
        Table.add_column(Networks[j + 1],
                         [item for sublist in val[:, j, Graph_Term] for item in sublist])
    print(
        '------------------ Batch Size - Classifier - Comparison - Accuracy  -----------------')
    print(Table)



def Plot_Results():
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR +', 'LR -', 'DOR', 'Prevalence']
    Graph_Term = [0, 4, 5, 6, 7, 8, 1, 3, 9, 10]  # Metrics to plot

    Eval = np.load('Eval_all_Act.npy', allow_pickle=True)

    # Classifiers and updated color palette
    Cls = ['Attention guidedDNN', 'CNN', 'Resnet', 'VGG19', 'GDD-GDAN']
    Colors_cls = ['#0072B2', '#E69F00', '#56B4E9', '#009E73', '#D55E00']  # vibrant & distinct colors

    os.makedirs("./Results", exist_ok=True)

    for m in range(len(Graph_Term)):
        Graph = Eval[:, 5:, Graph_Term[m] + 4]  # Shape: (num_classes, num_kfolds)
        bar_width = 0.15
        fig, ax = plt.subplots(figsize=(8, 6))

        x = np.arange(Graph.shape[1])  # x-axis for folds

        # Plot each classifier
        for i in range(len(Cls)):
            y_vals = Graph[:, i]
            bars = plt.bar(x + i * bar_width, y_vals, width=bar_width, color=Colors_cls[i], label=Cls[i], alpha=0.9)

            # Add marker (dot) on top of each bar
            for j, bar in enumerate(bars):
                height = bar.get_height()
                plt.plot(bar.get_x() + bar.get_width() / 2, height, marker='o',
                         markersize=6, color='black', markerfacecolor=Colors_cls[i])

        plt.xticks(x + bar_width * 2, ['Linear', 'ReLU', 'TanH', 'Sigmoid', 'Softmax'], fontsize=10)
        plt.xlabel('Activation Function', fontsize=12, fontweight='bold')
        plt.ylabel(Terms[Graph_Term[m]], fontsize=12, fontweight='bold')

        # Legend with dot markers
        dot_markers = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                       markeredgecolor='black', markersize=8)
            for color in Colors_cls
        ]
        plt.legend(dot_markers, Cls, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   fontsize=10, frameon=False, ncol=len(Cls))

        # Style cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        path = f"./Results/Classification_method_{Terms[Graph_Term[m]]}.png"
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

def Kfold_proposed():
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR +', 'LR -', 'DOR', 'Prevalence']
    Graph_Term = [0, 4, 5, 6, 7, 8, 1, 3, 9, 10]  # Metrics to plot

    Eval = np.load('Eval_all_Kfold.npy', allow_pickle=True)

    KFOLD = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    for m in range(len(Graph_Term)):
        Graph = Eval[:, 4, Graph_Term[m] + 4]  # Shape: (num_classes, num_kfolds)

        colors = [
            '#E74C3C', '#9B59B6', '#D4AC0D', '#7D6608', '#E67E22',
            '#AF7AC5', '#73C6B6', '#27AE60', '#F39C12', '#884EA0'
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Plot bars with tick labels directly
        x = np.arange(len(KFOLD))
        bars = ax.bar(x, Graph, color=colors, zorder=2)

        # Set ticks correctly (removes the warning)
        ax.set_xticks(x)
        ax.set_xticklabels(KFOLD)

        # Auto Y-ticks and alternating background bands
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        plt.draw()
        yticks = ax.get_yticks()
        plt.xlabel('Kfold', fontsize=12, fontweight='bold')
        plt.ylabel(Terms[Graph_Term[m]], fontsize=12, fontweight='bold')

        for i in range(len(yticks) - 1):
            color = '#f5f5f5' if i % 2 == 1 else 'white'
            ax.axhspan(yticks[i], yticks[i + 1], facecolor=color, alpha=1, zorder=0)

        # Labels above bars (2 decimal points)
        for bar, val in zip(bars, Graph):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (max(Graph) * 0.015),
                    f"{val:.2f}",  # <-- 2 digits only
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#555555', zorder=3)

        # Clean style
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')

        plt.tight_layout()
        path = f"./Results/KFOLD_PROPOSED_{Terms[Graph_Term[m]]}.png"
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


if __name__ == '__main__':
    # ROC_curve()
    # Plot_table()
    # Plot_Results()
    Plot_Confusion()

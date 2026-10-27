from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch



def plot_results_conv():
    conv = np.load('Fitness.npy', allow_pickle=True)

    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['NGO-ADLabV3+', 'DOA-ADLabV3+', 'AOA-ADLabV3+', 'SHVO-ADLabV3+', 'HISHVO-ADLabV3+']
    color = ['#C20078', '#00FFFF', '#A0522D', '#008080', '#006400']
    markerfacecolor = ['#13EAC9', '#FC5A50', '#C79FEF', '#C20078', '#000080']
    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("Statistical", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print('------------------------------------- Statistical Analysis -------------------------------------------')
    print(Table)

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence')
    iteration = np.arange(conv.shape[1])
    for m in range(conv.shape[0]):
        plt.plot(iteration, conv[m, :], color=color[m], linewidth=3, marker='h',
                 markerfacecolor=markerfacecolor[m], markersize=8, label=Algorithm[m])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    path = "./Results/Conv.png"
    plt.savefig(path)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def plot_results_seg_table():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
             'NPV',
             'FDR', 'F1-Score', 'MCC']
    Dataset = ['Dataset 1', 'Dataset 2', 'Dataset 3']

    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Full = ['TERMS', 'UNet', 'ResUNet', 'ICNN', 'DLabV3+', 'HISHVO-ADLabV3+']
    Alg = ['TERMS', 'NGO-ADLabV3+', 'DOA-ADLabV3+', 'AOA-ADLabV3+', 'SHVO-ADLabV3+', 'HISHVO-ADLabV3+']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            Table = PrettyTable()
            Table.add_column(Alg[0], Statistics)
            for k in range(5):
                Table.add_column(Alg[k + 1], stats[i, k, :])
            print('-------------------------------------------------- ', Dataset[n], Terms[i - 4],
                  'Algorthm  for Segmentation -------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Full[0], Statistics)
            Table.add_column(Full[1], stats[i, 5, :])
            Table.add_column(Full[2], stats[i, 6, :])
            Table.add_column(Full[3], stats[i, 7, :])
            Table.add_column(Full[4], stats[i, 8, :])
            Table.add_column(Full[5], stats[i, 4, :])
            print('-------------------------------------------------- ', Dataset[n], Terms[i - 4],
                  'Comparison for Segmentation -------------------------------------------------')
            print(Table)


def rounded_bar(ax, x, height, width=0.3, color='b', label=None, radius=0.05):
    """Draw a bar with rounded top corners and text value inside"""
    for xi, h in zip(x, height):
        # Draw the bar
        if h >= 0:
            bbox = FancyBboxPatch(
                (xi - width / 2, 0), width, h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                linewidth=0, facecolor=color
            )
            # Add text in the middle of the bar, rotated 90°, 6 decimal places, bold
            ax.text(xi, h / 2, f"{h:.6f}", ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold', rotation=90)
        else:
            bbox = FancyBboxPatch(
                (xi - width / 2, h), width, -h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                linewidth=0, facecolor=color
            )
            ax.text(xi, h / 2, f"{h:.6f}", ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold', rotation=90)

        ax.add_patch(bbox)

    if label is not None:
        ax.bar(0, 0, width, color=color, label=label)  # for legend


def Plot_Results_seg_method():
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
             'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2]
    Classifier = ['Linear', 'Relu', 'Tanh', 'Sigmoid', 'Softmax']

    Eval = np.load('Eval_Seg.npy', allow_pickle=True)

    bar_width = 0.30  # Width of each bar
    corner_radius = 0.05  # Rounded corner radius
    gap_between_groups = 0.5  # Space between clusters

    for j in range(len(Graph_Term)):
        Graph = Eval[:, :, Graph_Term[j] + 4]

        # Compute x positions with extra gap between clusters
        base_x = np.arange(len(Classifier)) * (5 * bar_width + gap_between_groups)

        # --- First set of bars (NGO, DOA, AOA, SHVO, PROPOSED) ---
        fig, ax = plt.subplots(figsize=(12, 6))


        rounded_bar(ax, base_x - 2 * bar_width, Graph[:, 0], width=bar_width, color='#13EAC9', label='NGO-ADLabV3+',
                    radius=corner_radius)
        rounded_bar(ax, base_x - bar_width, Graph[:, 1], width=bar_width, color='#FC5A50', label='DOA-ADLabV3+',
                    radius=corner_radius)
        rounded_bar(ax, base_x, Graph[:, 2], width=bar_width, color='#C79FEF', label='AOA-ADLabV3+', radius=corner_radius)
        rounded_bar(ax, base_x + bar_width, Graph[:, 3], width=bar_width, color='#C20078', label='SHVO-ADLabV3+',
                    radius=corner_radius)
        rounded_bar(ax, base_x + 2 * bar_width, Graph[:, 4], width=bar_width, color='#000080', label= 'HISHVO-ADLabV3+',
                    radius=corner_radius)

        ax.set_ylim(0, 1)
        ax.set_ylabel(Terms[Graph_Term[j]], fontsize=14, fontweight='bold', color='black')
        ax.set_xticks(base_x)
        ax.set_xticklabels(Classifier, fontweight='bold')
        ax.set_xlabel('Activation Function', fontweight='bold')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=True,
                  prop={'family': 'serif', 'size': 12})

        path = f"./Results/Activation_Seg_{Terms[Graph_Term[j]]}_Alg.png"
        plt.savefig(path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

        # --- Second set of bars (UNet, ResiUNet, ICNN, LabV3p, PROPOSED) ---
        fig, ax = plt.subplots(figsize=(12, 6))

        rounded_bar(ax, base_x - 2 * bar_width, Graph[:, 5], width=bar_width, color='#C20078', label='UNet',
                    radius=corner_radius)
        rounded_bar(ax, base_x - bar_width, Graph[:, 6], width=bar_width, color='#00FFFF', label='ResUNet',
                    radius=corner_radius)
        rounded_bar(ax, base_x, Graph[:, 7], width=bar_width, color='#A0522D', label='ICNN', radius=corner_radius)
        rounded_bar(ax, base_x + bar_width, Graph[:, 8], width=bar_width, color='#008080', label='DLabV3+',
                    radius=corner_radius)
        rounded_bar(ax, base_x + 2 * bar_width, Graph[:, 9], width=bar_width, color='#006400', label='HISHVO-ADLabV3+',
                    radius=corner_radius)

        ax.set_ylim(0, 1)
        ax.set_ylabel(Terms[Graph_Term[j]], fontsize=14, fontweight='bold', color='black')
        ax.set_xticks(base_x)
        ax.set_xticklabels(Classifier, fontweight='bold')
        ax.set_xlabel('Activation Function', fontweight='bold')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=True,
                  prop={'family': 'serif', 'size': 12})

        path = f"./Results/Activation_Seg_{Terms[Graph_Term[j]]}_Method.png"
        plt.savefig(path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


if __name__ == '__main__':
    plot_results_conv()
    Plot_Results_seg_method()
    plot_results_seg_table()

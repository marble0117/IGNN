import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy(name, show_list):
    path_b = 'result/baselines_' + name + '.csv'
    path_s = 'result/similarity_' + name + '.csv'
    df_b = pd.read_csv(path_b, index_col=0)
    df_s = pd.read_csv(path_s, index_col=0)
    df = pd.concat([df_b, df_s])
    df_index = df.index
    df_values = df.values
    x = [i*5 for i in range(1, 21)]
    color = ['g', 'r', 'y', 'c', 'm', 'b']
    color_dict = {cent: color[i] for i, cent in enumerate(show_list)}
    for i, idx in enumerate(df_index):
        idx_sp = idx.split(',')
        print(idx_sp)
        if idx_sp[0] in show_list and idx_sp[1] == 'avg':
            if idx_sp[2] == 'important':
                plt.plot(x, df_values[i], label=idx, marker='o', color=color_dict[idx_sp[0]])
            else:
                plt.plot(x, df_values[i], label=idx, marker='s', color=color_dict[idx_sp[0]], linestyle='--')
    plt.title(name)
    plt.ylabel("accuracy")
    plt.xlabel("eliminated edges (%)")
    plt.legend()
    plt.show()


def main():
    name = 'pubmed'
    show_list = ['neighbors', 'jaccard']
    plot_accuracy(name, show_list)

main()
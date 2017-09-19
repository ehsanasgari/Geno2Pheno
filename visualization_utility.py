import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def create_mat_plot(mat, axis_names, title, filename, cmap='inferno'):
    '''
    :param mat: divergence matrix
    :param axis_names: axis_names
    :param title
    :param filename: where to be saved
    :return:
    '''
    ax = sns.heatmap(mat,annot=False,yticklabels=axis_names,xticklabels=axis_names, cmap=cmap)
    plt.title(title)
    plt.savefig(filename + '.pdf')
    plt.clf()

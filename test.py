import matplotlib.pyplot as plt
from utils import yaml_utils
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties


def visualization_result(final_embeddings_path, reverse_dictionary_path, output_image, num_plot):
    def plot_with_labels(low_dim_embs, labels, filename, fonts):
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y, s=18)
            plt.annotate(label, fontproperties=fonts, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                         va='bottom')
        plt.savefig(filename)

    final_embeddings = yaml_utils.read(final_embeddings_path)
    reverse_dictionary = yaml_utils.read(reverse_dictionary_path)
    # 为了在图片上能显示出中文
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=18)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(final_embeddings[:num_plot, :])
    labels = [reverse_dictionary[i] for i in range(num_plot)]
    plot_with_labels(low_dim_embs, labels, output_image, fonts=font)


if __name__ == '__main__':
    dataset_name = 'wiki_corpus'
    model_name = 'SkipGram'
    tag = 'base'
    dataset_info = yaml_utils.read('dataset/' + dataset_name + '/info.yaml')
    checkpoint_dir = './_checkpoint/' + dataset_name + '/' + model_name + '/' + tag + '/'
    visualization_result(checkpoint_dir + 'final_embeddings.yaml', dataset_info['reverse_dictionary'],
                         checkpoint_dir + 'tsne.png', 100)

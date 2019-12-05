import matplotlib.pyplot as plt
import tensorflow as tf
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


def most_similar(final_embeddings_path, reverse_dictionary_path, words):
    final_embeddings = yaml_utils.read(final_embeddings_path)
    reverse_dictionary = yaml_utils.read(reverse_dictionary_path)
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        valid_dataset = tf.placeholder(tf.int32, shape=[1])
        normalized_embeddings = tf.placeholder(tf.float32, shape=[5000, 128])
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sim = sess.run(similarity, feed_dict={valid_dataset: words, normalized_embeddings: final_embeddings})
        valid_word = reverse_dictionary[words[0]]
        top_k = 5
        nearest = (-sim[0, :]).argsort()[1:top_k + 1]
        log_str = '最接近 ' + valid_word + ' 的词语是:'
        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str += str(k + 1) + ':' + close_word + ' '
        print(log_str)


if __name__ == '__main__':
    dataset_name = 'wiki_corpus'
    model_name = 'SkipGram'
    tag = 'base'
    dataset_info = yaml_utils.read('dataset/' + dataset_name + '/info.yaml')
    checkpoint_dir = './_checkpoint/' + dataset_name + '/' + model_name + '/' + tag + '/'
    # visualization_result(checkpoint_dir + 'final_embeddings.yaml', dataset_info['reverse_dictionary'],
    #                      checkpoint_dir + 'tsne.png', 100)

    most_similar(checkpoint_dir + 'final_embeddings.yaml', dataset_info['reverse_dictionary'], [62])

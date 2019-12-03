import math
import numpy as np
import random
import tensorflow as tf
from pathlib import Path

from tensorboard.plugins import projector

from data_loader import generate_batch
from utils import yaml_utils


class SkipGram:
    def __init__(self, dataset_name, batch_size, valid_size, valid_window, name, checkpoint_dir, embedding_size,
                 num_sampled, num_epoch, sess, tag, **args):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_info = yaml_utils.read(args['dataset']['path'])
        self.vocabulary_size = self.dataset_info['vocabulary_size']
        self.reverse_dictionary = yaml_utils.read(self.dataset_info['reverse_dictionary'])
        self.train_data_generator = generate_batch(self.dataset_info['data'], **args['dataset'])

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = valid_size
        self.valid_examples = np.array(random.sample(range(valid_window), valid_size))

        self.name = name
        self.checkpoint_dir = Path(checkpoint_dir) / self.dataset_name / self.name / tag
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled

        self.num_epoch = num_epoch
        self.valid_freq = 10000
        self.sess = sess
        self.tag = tag
        self.kwargs = args
        self.build_network()
        self.saver = tf.train.Saver()

    def build_network(self):
        # Input data.
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(self.embeddings, self.inputs)
            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                              stddev=1.0 / math.sqrt(self.embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]), dtype=tf.float32)

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=self.labels,
                               num_sampled=self.num_sampled, num_classes=self.vocabulary_size))
        loss_summary = tf.summary.scalar('loss', self.loss)
        self.scalar_summary = tf.summary.merge([loss_summary])
        # Compute the cosine similarity between minibatch examples and all embeddings. L2
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, valid_dataset)
        self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)

    def train(self):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
        # We must initialize all variables before we use them.
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        writer = tf.summary.FileWriter('../tensorboard_logs/{}/{}/{}'.format(self.dataset_name, self.name, self.tag),
                                       self.sess.graph)

        for epoch in range(self.num_epoch):
            batch_inputs, batch_labels = next(self.train_data_generator)
            _, loss_val, summary = self.sess.run([optimizer, self.loss, self.scalar_summary],
                                                 feed_dict={self.inputs: batch_inputs, self.labels: batch_labels})
            writer.add_summary(summary, epoch)
            if epoch % self.valid_freq == 0:
                sim = self.similarity.eval()
                print('>>第{}次验证:'.format(epoch))
                for i in range(self.valid_size):
                    valid_word = self.reverse_dictionary[self.valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = '最接近 ' + valid_word + ' 的词语是:'
                    for k in range(top_k):
                        close_word = self.reverse_dictionary[nearest[k]]
                        log_str += str(k + 1) + ':' + close_word + ' '
                    print(log_str)
                # save train model
                self.save(self.checkpoint_dir / 'train', self.saver, epoch)
        final_embeddings = self.normalized_embeddings.eval()
        # Write corresponding labels for the embeddings.
        self.save(self.checkpoint_dir / 'train', self.saver, self.num_epoch)
        yaml_utils.write(self.checkpoint_dir / 'final_embeddings.yaml', final_embeddings)
        # Create a configuration for visualizing embeddings with the labels in TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = self.embeddings.name
        embedding_conf.metadata_path = self.dataset_info['dictionary']
        projector.visualize_embeddings(writer, config)
        writer.close()

    def save(self, checkpoint_dir, saver, epoch, **kwargs):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        saver.save(self.sess, str(checkpoint_dir / 'model.cpk'), global_step=epoch)

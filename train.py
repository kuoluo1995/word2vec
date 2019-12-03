import tensorflow as tf
from model import SkipGram
from utils.config_utils import get_config


def train(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = SkipGram(sess=sess, **args['dataset'], **args['model'], **args)
        model.train()


if __name__ == '__main__':
    config = get_config('base')
    config['tag'] = 'base'
    train(config)

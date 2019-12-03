import collections
import numpy as np
import random

from utils import yaml_utils


def generate_batch(data_path, batch_size, num_skips, skip_window, **args):
    data = yaml_utils.read(data_path)
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    buffer.extend(data[0:span])
    data_index = span  # init buffer and data_index
    while True:
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if data_index == len(data):
                buffer.extend(data[0:span])
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data)
        yield batch, labels


if __name__ == '__main__':
    print('读取数据')
    dataset = yaml_utils.read('dataset/little_data.yaml')
    print('读取数据完毕')
    data = dataset['data']
    reverse_dictionary = dataset['reverse_dictionary']
    generator = generate_batch(data, 8, 2, 1)
    batch, labels = next(generator)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

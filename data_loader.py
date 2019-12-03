import collections
import numpy as np
import random

from utils import yaml_utils


def generate_batch(data_path, batch_size, num_skips, skip_window, **args):
    data = yaml_utils.read(data_path)
    data_index = 0
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    while True:
        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]

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

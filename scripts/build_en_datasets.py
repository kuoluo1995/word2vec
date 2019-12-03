import collections
from pathlib import Path
from utils import yaml_utils

vocabulary_size = 5000
dataset_name = 'text8'  # wiki_corpus
output_dir = Path('../dataset').absolute()


def read_data(path):
    data_path = Path(path)
    raw_word_list = list()
    total = 1
    with data_path.open(mode='r', encoding='UTF-8') as f:
        line = f.readline()
        while line:
            line = f.readline()
            total += 1
            print('\r当前第{}行'.format(total), end='')
    with data_path.open(mode='r', encoding='UTF-8') as f:
        line = f.readline()
        i = 1
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            if len(line) > 0:
                raw_words = list(line.split())
                raw_word_list.extend(raw_words)
            print('\r >>当前读取到{}/{}行'.format(i, total), end='')
            line = f.readline()
            i += 1
    return raw_word_list


# Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(words):
    print('统计字符出现的数量')
    count = [('UNK', -1)]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print('创建词典')
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    print('根据词典转化原数据成序列')
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0] = ('UNK', unk_count)
    print('制作反向查询词典')
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    output_path = output_dir / dataset_name
    print('导出文件')
    yaml_utils.write(output_path / 'data.yaml', data)
    yaml_utils.write(output_path / 'dictionary.yaml', dictionary)
    yaml_utils.write(output_path / 'reverse_dictionary.yaml', reverse_dictionary)
    with (output_path / 'dictionary.tsv').open(mode='w', encoding='UTF-8') as file:
        for i in range(vocabulary_size):
            file.write(reverse_dictionary[i] + '\n')
    info_dict = {'vocabulary_size': vocabulary_size, 'data': str(output_path / 'data.yaml'),
                 'dictionary': str(output_path / 'dictionary.tsv'),
                 'reverse_dictionary': str(output_path / 'reverse_dictionary.yaml')}
    yaml_utils.write(output_path / 'info.yaml', info_dict)
    print('导出完成')


if __name__ == '__main__':
    words = read_data('../data/{}.txt'.format(dataset_name))
    build_dataset(words)

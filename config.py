import warnings
import os


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'HBT'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    root_path = os.path.dirname(os.path.realpath(__file__))

    train_data_root = './data/train_triples.json'  # 训练集存放路径
    dev_data_root = './data/dev_triples.json'  # 验证集存放路径
    test_data_root = './data/test_triples.json'  # 测试集存放路径
    rel_data = './data/rel2id.json'  # 关系数据路径

    bert_config_file = './checkpoints/cased_L-12_H-768_A-12/bert_config.json'
    bert_vocab_file = './checkpoints/cased_L-12_H-768_A-12/vocab.txt'
    bert_pretrained_model = './checkpoints/cased_L-12_H-768_A-12/pytorch_model.bin'  # 加载预训练的模型的路径，为None代表不加载
    # bert_pretrained_model = None
    output_dir = './checkpoints/my_model'
    load_model_name = 'pytorch_model_last'
    output_path = './checkpoints/dev_result/res'

    train = True
    dev = False
    test = False

    batch_size = 64  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 10  # print info every N batch
    save_freq = 1000  # save model every save_freq steps
    eval_freq = 1000  #
    max_len = 100  # the max length of sentence

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 3
    lr = 1e-5  # initial learning rate
    warmup_proportion = 0.1

    hidden_size = 768
    rel_nums = 24
    initializer_range = 0.02


def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    # 更新配置参数
    for k, v in kwargs.iteritems():
        if not hasattr(self, k):
            # 警告还是报错，取决于你个人的喜好
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self, k, v)

    # 打印配置信息
    print('user config:')
    for k, v in self.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k, getattr(self, k))
import warnings

from numpy.lib.function_base import gradient


class Config(object):
    # DATA
    dataset = 'kinetics400'
    num_class = 400
    train_data_root = r'/home/sjhu/projects/kinetics400-mpeg4'
    test_data_root = r'/home/sjhu/projects/kinetics400-mpeg4'
    train_list = r'/home/sjhu/projects/k400_train.txt'
    test_list = r'/home/sjhu/projects/k400_val.txt'


    # MODEL
    num_segments = 5
    alpha = 2  # ratio of mv/frame sample rate
    model = ''

    # TRAIN
    epochs = 300
    grad_accumu_steps = 1
    batch_size = 8
    weight_decay = 1e-4
    workers = 8
    lr = 0.01
    lr_steps = [120,200,280]
    lr_decay = 0.1
    eval_freq = 5

    #TEST
    test_crops = 1
    

    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('------------------------ user config: ------------------------------')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
        print('-----------------------  ************ ------------------------------')

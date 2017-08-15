import logging

import mxnet as mx

# 读取图片,读取数据之前先在开始处设定data_dir
import logging
import os

import cv2
import mxnet as mx
import numpy as np
from logger import logger

stars_dataset = False  # 是否使用facescrub数据集
data_dir = '/home/haowei/face/megaface_tight/'
# data_dir = '/Users/haowei/facerecog/face/megaface_tight'
data_dir = '../megaface_tight'
test_data_dir = '../testface'
binary_train = True  # 是否二分类

with open('./persons.txt', 'r') as f:
    names = list(map(lambda s: s.strip(), f.readlines()))
    print(names)

'''
    read in pic
'''


def parse_dir(filenames_list):
    files_list = []
    for name in filenames_list:
        try:
            p = cv2.imread(name)
            p = cv2.resize(p, (144, 144))
            '''
            cv2.imshow('p',p)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            p = np.stack([p[:,:,i] for i in range(3)],axis=0)
            files_list.append(p)
        except Exception as e:
            logger.info(e)
            logger.info(name)
    return files_list


def read_in():
    cropus_filename = []
    for i, p in enumerate(os.listdir(data_dir)):
        p = os.path.join(data_dir, p)
        cropus_filename.append(list(map(lambda f: os.path.join(p, f), os.listdir(p))))
    cropus_data = []

    for filenames_list in cropus_filename:
        temp = parse_dir(filenames_list)
        cropus_data.append(temp)
    logging.info("read file in")
    return cropus_data


cropus_data = read_in()
cropus_data_cnn = cropus_data
global cropus

'''
    划分数据集
'''


def parse(len_of_test, test_person_id):
    n = np.array(list((map(lambda s: len(s), cropus_data))))

    global cropus
    cropus = {}
    for i in ['train_data', 'train_label', 'test_data', 'test_label']:
        cropus[i] = []

    for i, d in enumerate(cropus_data_cnn):
        test_idx = int(len(d) / 8)
        train_num = len(d) - test_idx
        '''
        if i == index_test:
            cropus['train_label'] += [int(i == index_test)] * len_of_test
            cropus['test_label'] += [int(i == index_test)] * (len(d) - len_of_test)
            cropus['test_data'] += d[len_of_test:]
            cropus['train_data'] += d[0:len_of_test]
        else:
        '''
        cropus['train_label'] += [0] * train_num
        cropus['test_label'] += [0] * test_idx
        cropus['train_data'] += d[test_idx:]
        cropus['test_data'] += d[0:test_idx]

    temp = list(map(lambda s:os.path.join(test_data_dir,names[test_person_id],s),os.listdir(os.path.join(test_data_dir, names[test_person_id]))))
    test_cropus = parse_dir(temp)
    # 用3个做val集
    len_of_val = 3
    cropus['test_label'] += [1] *(len_of_val)
    cropus['test_data'] += test_cropus[:len_of_val]
    cropus['train_label'] += [1]*(len_of_test-len_of_val)
    cropus['train_data'] += test_cropus[len_of_val:len_of_test]

    for i in ['train_data', 'train_label', 'test_data', 'test_label']:
        logger.info(len(cropus[i]))
        cropus[i] = np.asarray(cropus[i])
    logger.info('train postive number:{},test postive number {}'.format(sum(cropus['train_label']),
                                                                        sum(cropus['test_label'])))
    logger.info(sum(cropus['test_label']) / len(cropus['test_label']))

def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec='./caltech-256-60-train.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        shuffle=True,
        rand_crop=True,
        rand_mirror=True)
    val = mx.io.ImageRecordIter(
        path_imgrec='./caltech-256-60-val.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        rand_crop=False,
        rand_mirror=False)
    return (train, val)


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)


head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)


def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
            num_epoch=8,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            batch_end_callback=mx.callback.Speedometer(batch_size, 10),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)


num_classes = 2
batch_per_gpu = 16
num_gpus = 1
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)
(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)

def parse_train_and_eval(len_of_test, result, test_person_id, epochs_num):
    global cropus
    parse(len_of_test=len_of_test, test_person_id=test_person_id)
    batch_size = 64
    train_iter = mx.io.NDArrayIter(cropus['train_data'], cropus['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(cropus['test_data'], cropus['test_label'], batch_size)
    batch_size = batch_per_gpu * num_gpus
    fit(new_sym, new_args, aux_params, train_iter, val_iter, batch_size, num_gpus)

def train_all_model(epochs_num,len_of_test = 30):
    print(names)
    result = {}
    for i in range(len(names)):
        logging.info('{}:{}'.format(i,names[i]))
        parse_train_and_eval(len_of_test=len_of_test, result=result, test_person_id=i,epochs_num = epochs_num)
    pass
    logger.info(result)

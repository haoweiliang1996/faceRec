# 读取图片,读取数据之前先在开始处设定data_dir
import logging
import os

import cv2
import mxnet as mx
import numpy as np
from logger import logger

stars_dataset = False  # 是否使用facescrub数据集
data_dir = '../megaface_tight'
mean_image = mx.nd.load('mean.ndarray')['mean_image'].asnumpy()
'''
    read in pic
'''


def parse_dir(filenames_list):
    files_list = []
    for name in filenames_list:
        try:
            temp = cv2.imread(name)
            temp = cv2.resize(temp, (224, 224))
            temp = np.stack([temp[:,:,i] for i in range(3)],axis=0)
            p = temp - mean_image
            files_list.append(p)
        except Exception as e:
            logger.info(e)
            logger.info(name)
    return files_list


def read_in():
    if os.path.exists("resnet-101-training-data.npy"):
        logger.info("load exist data")
        #return np.load("resnet-101-training-data.npy")
    cropus_filename = []
    for i, p in enumerate(os.listdir(data_dir)):
        p = os.path.join(data_dir, p)
        cropus_filename.append(list(map(lambda f: os.path.join(p, f), os.listdir(p))))
    cropus_data = []

    for filenames_list in cropus_filename:
        temp = parse_dir(filenames_list)
        cropus_data.append(temp)
    logging.info("read file in")
    #np.save("resnet-101-training-data",cropus_data)
    return cropus_data


cropus_data = read_in()
cropus_data_cnn = cropus_data
global cropus

'''
    划分数据集
'''


def parse():
    global cropus
    cropus = {}
    for i in ['train_data', 'train_label', 'test_data', 'test_label']:
        cropus[i] = []

    for i, d in enumerate(cropus_data_cnn):
        test_idx = int(len(d) / 5)
        train_num = len(d) - test_idx
        cropus['train_label'] += [i + 10576] * train_num
        cropus['test_label'] += [i + 10576] * test_idx
        cropus['train_data'] += d[test_idx:]
        cropus['test_data'] += d[0:test_idx]


    for i in ['train_data', 'train_label', 'test_data', 'test_label']:
        logger.info(len(cropus[i]))
        cropus[i] = np.asarray(cropus[i])

    logger.info(sum(cropus['test_label']) / len(cropus['test_label']))

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten_0'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    print(all_layers.list_outputs()[-10:])
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)




def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    num_epoch = 50
    mod.fit(train, val,
        num_epoch=num_epoch,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 1000),
        optimizer='adam',
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc',
        epoch_end_callback=mx.callback.do_checkpoint('fine-tuned',num_epoch))
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)

num_classes = len(cropus_data)
num_gpus = 1
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-101', 0)
(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)

def parse_train_and_eval():
    global cropus
    parse()
    batch_size = 32
    train_iter = mx.io.NDArrayIter(cropus['train_data'], cropus['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(cropus['test_data'], cropus['test_label'], batch_size)
    fit(new_sym, new_args, aux_params, train_iter, val_iter, batch_size,num_gpus)

parse_train_and_eval()

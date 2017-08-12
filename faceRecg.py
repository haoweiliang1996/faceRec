# 读取图片,读取数据之前先在开始处设定data_dir
from collections import namedtuple
import os
import mxnet as mx
import cv2
import numpy as np
import logging

stars_dataset = False  # 是否使用facescrub数据集
if not stars_dataset:
    data_dir = '/home/haowei/face/megaface_tight/'
    # data_dir = '/Users/haowei/facerecog/face/megaface_tight'
else:
    data_dir = '/home/haowei/face/faceRec/facescrub_aligned_100/'
binary_train = True  # 是否二分类

with open('./persons.txt', 'r') as f:
    names = list(map(lambda s: s.strip(), f.readlines()))

'''
    read in pic
'''


def read_in():
    cropus_filename = []
    for i, p in enumerate(os.listdir(data_dir)):
        p = os.path.join(data_dir, p)
        cropus_filename.append(list(map(lambda f: os.path.join(p, f), os.listdir(p))))
    cropus_data = []

    def parse_dir(filenames_list):
        files_list = []
        for name in filenames_list:
            try:
                p = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
                p = cv2.resize(p, (64, 64))
                '''
                cv2.imshow('p',p)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
                p = p[np.newaxis, :]
                files_list.append(p)
            except Exception as e:
                print(e)
                print(name)
        return files_list

    for filenames_list in cropus_filename:
        temp = parse_dir(filenames_list)
        cropus_data.append(temp)
    logging.error("read file in")
    return cropus_data


cropus_data = read_in()
cropus_data_cnn = cropus_data
global cropus


def parse(len_of_test, test_person_id):
    for i, dir_name in enumerate(os.listdir(data_dir)):
        if dir_name == names[test_person_id]:
            index_test = i
    n = np.array(list((map(lambda s: len(s), cropus_data))))
    print(n[index_test])

    global cropus
    cropus = {}
    for i in ['train_data', 'train_label', 'test_data', 'test_label']:
        cropus[i] = []

    for i, d in enumerate(cropus_data_cnn):
        test_idx = int(len(d) / 5 + 1)
        train_num = len(d) - test_idx
        if i == index_test:
            cropus['train_label'] += [int(i == index_test)] * len_of_test
            cropus['test_label'] += [int(i == index_test)] * (len(d) - len_of_test)
            cropus['test_data'] += d[len_of_test:]
            cropus['train_data'] += d[0:len_of_test]
        else:
            cropus['train_label'] += [int(i == index_test)] * train_num
            cropus['test_label'] += [int(i == index_test)] * test_idx
            cropus['train_data'] += d[test_idx:]
            cropus['test_data'] += d[0:test_idx]
    for i in ['train_data', 'train_label', 'test_data', 'test_label']:
        print(len(cropus[i]))
        cropus[i] = np.asarray(cropus[i])
    print('train postive number:{},test postive number {}'.format(sum(cropus['train_label']),
                                                                  sum(cropus['test_label'])))
    print(sum(cropus['test_label']) / len(cropus['test_label']))


def parse_train_and_eval(len_of_test, result, test_person_id,epochs_num):
    global cropus
    parse(len_of_test=len_of_test, test_person_id=test_person_id)
    batch_size = 256
    train_iter = mx.io.NDArrayIter(cropus['train_data'], cropus['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(cropus['test_data'], cropus['test_label'], batch_size)

    def get_model():
        data = mx.sym.var('data')
        class_num = len(cropus_data_cnn)
        if binary_train:
            class_num = 2
        print('class_num {}'.format(class_num))
        # first conv layer
        conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=25)
        bn1 = mx.sym.BatchNorm(data=conv1)
        tanh1 = mx.sym.Activation(data=bn1, act_type="relu")
        pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2, 2), stride=(2, 2))
        # second conv layer
        conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=60)
        bn2 = mx.sym.BatchNorm(data=conv2)
        tanh2 = mx.sym.Activation(data=bn2, act_type="relu")
        pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2, 2), stride=(2, 2))
        # third conv layer
        conv3 = mx.sym.Convolution(data=pool2, kernel=(5, 5), num_filter=128)
        bn3 = mx.sym.BatchNorm(data=conv3)
        tanh3 = mx.sym.Activation(data=bn3, act_type="relu")
        pool3 = mx.sym.Pooling(data=tanh3, pool_type="max", kernel=(2, 2), stride=(2, 2))
        flatten = mx.sym.flatten(data=pool3)

        # remove third conv layer
        # flatten = mx.sym.flatten(data=pool2)

        # first fullc layer
        fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
        relu3 = mx.sym.Activation(data=fc1, act_type="relu")
        dropout1 = mx.symbol.Dropout(data=relu3, p=0.5)

        # second fullc layer
        fc11 = mx.symbol.FullyConnected(data=dropout1, num_hidden=500)
        relu4 = mx.sym.Activation(data=fc11, act_type="relu")
        dropout2 = mx.symbol.Dropout(data=relu4, p=0.5)

        # remove second fullc layer
        # dropout2 = dropout1
        # third fullc layer
        fc12 = mx.symbol.FullyConnected(data=dropout2, num_hidden=500)
        relu5 = mx.sym.Activation(data=fc12, act_type="relu")
        dropout3 = mx.symbol.Dropout(data=relu5, p=0.5)

        fc2 = mx.sym.FullyConnected(data=dropout3, num_hidden=class_num)
        # softmax loss
        lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
        return lenet

    # train
    def train(lenet):
        if binary_train:
            model_prefix = 'face_cnn_binary_{}'.format(test_person_id)
        else:
            model_prefix = 'face_cnn'

        checkpoint = mx.callback.do_checkpoint(model_prefix, epochs_num)
        eval_metrics = mx.metric.CompositeEvalMetric()
        if binary_train:
            eval_metrics.add(mx.metric.F1())
        else:
            eval_metrics.add(mx.metric.Accuracy())
        eval_metrics.add(mx.metric.Accuracy())

        import logging
        logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
        # create a trainable module on GPU 0
        lenet_model = mx.mod.Module(symbol=lenet, context=mx.gpu())
        # train with the same
        lenet_model.fit(train_iter,
                        eval_data=val_iter,
                        optimizer='Adadelta',
                        eval_metric=eval_metrics,
                        batch_end_callback=mx.callback.Speedometer(batch_size, frequent=50),
                        num_epoch=epochs_num, epoch_end_callback=checkpoint)
        # eval
        from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
        test_iter = mx.io.NDArrayIter(cropus['test_data'], cropus['test_label'], batch_size)
        prob = lenet_model.predict(test_iter)
        y_scores = prob.asnumpy()[:, 1]
        y_pred = []
        for i in prob:
            y_pred.append(i.asnumpy().argmax())
        y_labels = cropus['test_label']

        precision, recall, _ = precision_recall_curve(y_labels, y_scores)
        # print(list(y_scores))
        acc = mx.metric.Accuracy()
        lenet_model.score(test_iter, acc)
        print('acc {}'.format(acc))
        print('f1 {}'.format(f1_score(y_labels, y_pred)))
        print('p {}'.format(precision_score(y_labels, y_pred)))
        print('recall {}'.format(recall_score(y_labels, y_pred)))
        result[len_of_test] = [acc, f1_score(y_labels, y_pred), precision_score(y_labels, y_pred),
                               recall_score(y_labels, y_pred)]
        del lenet_model

    train(get_model())


def train_all_model(epochs_num,len_of_test = 30):
    for i in range(len(names)):
        logging.error(names[i])
        parse_train_and_eval(len_of_test=len_of_test, result=result, test_person_id=i,epochs_num = epochs_num)
    pass




result = {}
from pprint import pprint

pprint(result)

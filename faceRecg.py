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


def parse_train_and_eval(len_of_test, result, test_person_id, epochs_num):
    global cropus
    parse(len_of_test=len_of_test, test_person_id=test_person_id)
    batch_size = 64
    train_iter = mx.io.NDArrayIter(cropus['train_data'], cropus['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(cropus['test_data'], cropus['test_label'], batch_size)

    def get_model():
        data = mx.sym.var('data')
        class_num = len(cropus_data_cnn)
        if binary_train:
            class_num = 2
        logger.info('class_num {}'.format(class_num))
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

    def train(lenet):
        '''
        if binary_train:
            model_prefix = 'face_cnn_binary_{}'.format(test_person_id)
        else:
            model_prefix = 'face_cnn'

        checkpoint = mx.callback.do_checkpoint(model_prefix, epochs_num)
        '''
        eval_metrics = mx.metric.CompositeEvalMetric()
        eval_metrics.add(mx.metric.Accuracy())
        eval_metrics.add(mx.metric.F1())

        # create a trainable module on GPU 0
        lenet_model = mx.mod.Module(symbol=lenet, context=mx.gpu())
        lenet_model.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)
        lenet_model.init_params(initializer=mx.init.Uniform(scale=.1))
        lenet_model.init_optimizer(optimizer='Adadelta')

        best_acc = -1
        patience = 5
        pa_count = patience
        for epoch in range(50):
            train_iter.reset()
            eval_metrics.reset()
            for batch in train_iter:
                lenet_model.forward(batch,is_train=True)
                lenet_model.update_metric(eval_metrics,batch.label)
                lenet_model.backward()
                lenet_model.update()
            logger.info('Epoch {},Training {}'.format(epoch,eval_metrics.get()))
            score = lenet_model.score(val_iter,['acc','f1'])
            #score = eval_metrics.get()
            # print(score)

            logger.info('val acc {},f1 {}'.format(score[0][1],score[1][1]))
            if best_acc < score[0][1]:
                arg_params,aux_params = lenet_model.get_params()
                best_acc = score[0][1]
                pa_count = patience
                logger.info('best val acc get {}'.format(best_acc))
            elif best_acc == score[0][1]:
                arg_params,aux_params = lenet_model.get_params()
            elif best_acc > score[0][1]:
                pa_count -= 1
            if  pa_count< 0:
                break
        mx.model.save_checkpoint(prefix="face-cnn-person-{}".format(test_person_id),epoch = 0,symbol=lenet,arg_params=arg_params,aux_params=aux_params)


        # train with the same
        '''
        lenet_model.fit(train_iter,
                        eval_data=val_iter,
                        optimizer='Adadelta',
                        eval_metric=eval_metrics,
                        batch_end_callback=mx.callback.Speedometer(batch_size, frequent=50),
                        num_epoch=epochs_num, epoch_end_callback=checkpoint)
        '''
        # eval
        best_val_model = mx.mod.Module(symbol=lenet, context=mx.gpu())
        best_val_model.bind(for_training=False,data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)
        best_val_model.set_params(arg_params,aux_params)
        from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
        test_iter = mx.io.NDArrayIter(cropus['test_data'], cropus['test_label'], batch_size)
        prob = best_val_model.predict(test_iter)
        y_scores = prob.asnumpy()[:, 1]
        y_pred = []
        for i in prob:
            y_pred.append(i.asnumpy().argmax())
        y_labels = cropus['test_label']

        precision, recall, _ = precision_recall_curve(y_labels, y_scores)
        # logger.info(list(y_scores))
        acc = mx.metric.Accuracy()
        best_val_model.score(test_iter, acc)
        logger.info('acc {}'.format(acc))
        logger.info('f1 {}'.format(f1_score(y_labels, y_pred)))
        logger.info('p {}'.format(precision_score(y_labels, y_pred)))
        logger.info('recall {}'.format(recall_score(y_labels, y_pred)))
        result[test_person_id] = ([acc, f1_score(y_labels, y_pred), precision_score(y_labels, y_pred),
                                recall_score(y_labels, y_pred)])
        del lenet_model,best_val_model

    train(get_model())


def train_all_model(epochs_num,len_of_test = 30):
    print(names)
    result = {}
    for i in range(len(names)):
        logging.info('{}:{}'.format(i,names[i]))
        parse_train_and_eval(len_of_test=len_of_test, result=result, test_person_id=i,epochs_num = epochs_num)
    pass
    logger.info(result)




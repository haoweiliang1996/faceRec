# 读取图片,读取数据之前先在开始处设定data_dir
import logging
from logistic_regression import get_model
import os

import cv2
import mxnet as mx
import numpy as np
from logger import logger
from collections import namedtuple

stars_dataset = False  # 是否使用facescrub数据集
data_dir = '../megaface_tight'
test_data_dir = '../testface'
binary_train = True  # 是否二分类

with open('./persons.txt', 'r') as f:
    names = list(map(lambda s: s.strip(), f.readlines()))
    print(names)

'''
    read in pic
'''

mean_image = mx.nd.load('mean.ndarray')['mean_image']
def parse_dir(filenames_list):
    files_list = []
    for name in filenames_list:
        try:
            p = cv2.imread(name)
            p = cv2.resize(p, (224, 224))
            '''
            cv2.imshow('p',p)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            p = np.stack([p[:,:,i] for i in range(3)],axis=0)
            pic = p - mean_image.asnumpy()
            pic = pic[np.newaxis,:]
            files_list.append(pic)
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



sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-101', 0)
all_layers = sym.get_internals()
fe_sym = all_layers['flatten_0_output']
fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.gpu(), label_names=None)
fe_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
fe_mod.set_params(arg_params, aux_params)
Batch = namedtuple('Batch', ['data'])
def get_feature(img):
    fe_mod.forward(Batch([mx.nd.array(img)]))
    features = fe_mod.get_outputs()[0].asnumpy()
    print(features)
    return features

if __name__ == '__main__':
    train_crpous = cropus_data[0:300]
    def fill(crpous):
        datas = []
        labels = []
        for object in crpous:
            object_temp = object[:3]
            for id1,pic1 in enumerate(object_temp):
                for id2,pic2 in enumerate(object_temp):
                    datas.append(np.hstack((get_feature(pic1),get_feature(pic2))))
                    labels.append(int(id1==id2))
        return datas,labels
    train_datas,train_labels = fill(train_crpous)
    val_cropus= cropus_data[300:500]
    val_datas,val_labels = fill(val_cropus)
    logger.info("train len {},val len {}".format(len(train_labels),len(val_datas)))

    model = get_model()
    batch_size = 256
    train_iter = mx.io.NDArrayIter(train_datas,train_labels,batch_size,shuffle=True)
    val_iter= mx.io.NDArrayIter(val_datas,val_labels,batch_size)
    mod = mx.mod.Module(symbol=model,context=mx.gpu())
    mod.fit(train_iter,val_iter,optimizer="Adadelta",eval_metric='acc'
            ,batch_end_callback=mx.callback.Speedometer(batch_size,100),num_epoch=10)

'''
from sklearn.metrics.pairwise import cosine_similarity
a = get_feature(cropus_data[0][0])
b = get_feature(cropus_data[0][1])
c = get_feature(cropus_data[0][2])
d = get_feature(cropus_data[1][0])
print(cosine_similarity(a,b))
print(cosine_similarity(c,b))
print(cosine_similarity(a,c))
print(cosine_similarity(a,d))
'''

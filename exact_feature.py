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
binary_train = True  # 是否二分类

with open('./persons.txt', 'r') as f:
    names = list(map(lambda s: s.strip(), f.readlines()))
    print(names)

'''
    read in pic
'''

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
    return features[0]

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
            files_list.append(get_feature(pic))
        except Exception as e:
            logger.info(e)
            logger.info(name)
    return files_list


def read_in(data_dir):
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


def dump_features(data_dir,feature_name):
    cropus_data = read_in(data_dir)
    np.save(feature_name,cropus_data)

#dump_features('../megaface_tight',"face_feature")
'''
    划分数据集
'''
from sklearn.metrics.pairwise import cosine_similarity
if __name__ == '__main__':
    cropus_data = np.load("face_feature.npy")
    train_crpous = cropus_data[20:120]
    def fill(cropus):
        datas = {}
        # datas['img1'] = []
        # datas['img2'] = []
        datas['sim'] = []
        labels = []
        for id1,object1 in enumerate(cropus):
            object1 = object1[:10]
            for id2,object2 in enumerate(cropus):
                object2 = object2[:10]
                for pic1 in object1:
                    fe1 = pic1
                    for pic2 in object2:
                        fe2 = pic2
                        #print(temp[0].shape)
                        #print(np.array(cosine_similarity(fe1,fe2)[0]))
                        # datas.append(np.hstack((temp[0],np.array(cosine_similarity(fe1,fe2)[0]))[0]))
                        # datas['img1'].append(fe1)
                        # datas['img2'].append(fe2)
                        datas['sim'].append(np.array(cosine_similarity(fe1[np.newaxis,:],fe2[np.newaxis,:])[0]))
                        labels.append(int(id1==id2))
        return datas,mx.nd.array(labels)
    train_datas,train_labels = fill(train_crpous)
    val_cropus= cropus_data[120:135]
    val_datas,val_labels = fill(val_cropus)
    logger.info("train len {},val len {}".format(len(train_labels.asnumpy()),len(val_labels.asnumpy())))

    model = get_model()
    batch_size = 512
    logger.info('prepare data')
    train_iter = mx.io.NDArrayIter(train_datas,train_labels,batch_size,shuffle=True)
    val_iter= mx.io.NDArrayIter(val_datas,val_labels,batch_size)
    logger.info('begin train')
    mod = mx.mod.Module(symbol=model,context=mx.gpu(),data_names=['sim'])
    mod.fit(train_iter,val_iter,optimizer="adam",eval_metric='f1'
            ,batch_end_callback=mx.callback.Speedometer(batch_size,1000),epoch_end_callback=mx.callback.do_checkpoint('regression',150),num_epoch=150)

'''
a = get_feature(cropus_data[0][0])
b = get_feature(cropus_data[0][1])
c = get_feature(cropus_data[0][2])
d = get_feature(cropus_data[1][0])
print(cosine_similarity(a,b))
print(cosine_similarity(c,b))
print(cosine_similarity(a,c))
print(cosine_similarity(a,d))
'''

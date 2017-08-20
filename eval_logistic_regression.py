from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import namedtuple
from exact_feature import get_feature
from logger import logger
import mxnet as mx
import os
import cv2
data_dir = '../testface'
with open('./persons.txt', 'r') as f:
    names = list(map(lambda s: s.strip(), f.readlines()))
    print(names)
class facemodel():
    def __init__(self):
        sym, arg_params, aux_params = mx.model.load_checkpoint('regression', 150)
        mod = mx.mod.Module(symbol=sym, context=mx.gpu(), data_names=['sim'],label_names=None)
        mod.bind(for_training=False, data_shapes=[('sim', (1, 1))],
                 label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params, allow_missing=True)
        self.regression_model =mod
        self.models_list = []
        for i in range(len(names)):
            if names[i] == 'other':
                continue
        self.face_eval_data = []
        mean_image = mx.nd.load('mean.ndarray')['mean_image']
        if not os.path.exists("face_feature_test.npy"):
            logger.info('create test feature file')
            for name in names:
                pathname = os.path.join(data_dir, name)
                temp = []
                try:
                    for pic_name in os.listdir(pathname):
                        pic_name = os.path.join(pathname,pic_name)
                        p = cv2.imread(pic_name)
                        p = cv2.resize(p, (224,224))
                        p = np.stack([p[:,:,i] for i in range(3)],axis=0)
                        pic = p -mean_image.asnumpy()
                        pic = pic[np.newaxis, :]
                        temp.append(get_feature(pic))
                    self.face_eval_data.append(temp)
                except Exception as e:
                    logger.error(e)
            np.save("face_feature_test",self.face_eval_data)
        else:
            logger.info('load test feature file')
            self.face_eval_data = np.load("face_feature_test.npy")
        self.models_list = [i[:10] for i in self.face_eval_data]

    def predict(self, pic):
        Batch = namedtuple('Batch', ['data'])
        probs = None
        for fe1s in self.models_list:
            t = None
            mmax = None
            for fe1 in fe1s:
                fe = cosine_similarity(fe1,pic)[0]

                fe = fe[np.newaxis,:]
                self.regression_model.forward(Batch([mx.nd.array(fe)]))
                mmax = self.regression_model.get_outputs()[0].asnumpy()
                if t is None:
                    t = mmax
                elif t[0][1] < mmax[0][1]:
                    t = mmax
            # t /= len(fe1s)
            if probs is None:
                probs = t
            else:
                probs = np.vstack((probs, t))
        res = probs[:, 1].argmax()
        if probs[:, 1][res] < 0.5:
            return -1
        else:
            return res


from sklearn.metrics import accuracy_score

'''
    使用测试的数据集predict
'''


def eval_all_model():
    face_model = facemodel()
    y_true = []
    y_pred = []
    y_pred_without_other = []
    y_true_without_other = []
    for id, faces in enumerate(face_model.face_eval_data):
        logger.info('eval {} begin'.format(id))
        y_true += [id] * (len(faces) -10)
        for face in faces[10:]:
            temp = face_model.predict(pic=face)
            if temp != -1:
                y_pred_without_other.append(temp)
                y_true_without_other.append(id)
            y_pred.append(temp)
        logger.info('eval {} end'.format(id))
    logger.info(accuracy_score(y_true, y_pred))
    logger.info(accuracy_score(y_true_without_other, y_pred_without_other))
    pass

eval_all_model()

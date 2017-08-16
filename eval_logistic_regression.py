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
        sym, arg_params, aux_params = mx.model.load_checkpoint('regression', 2050)
        mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (1, 1))],
                 label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params, allow_missing=True)
        self.regression_model =mod
        self.models_list = []
        for i in range(len(names)):
            if names[i] == 'other':
                continue
        self.face_eval_data = []
        for name in names:
            pathname = os.path.join(data_dir, name)
            temp = []
            try:
                for pic_name in os.listdir(pathname):
                    pic_name = os.path.join(pathname,pic_name)
                    p = cv2.imread(pic_name)
                    p = cv2.resize(p, (224,224))
                    p = np.stack([p[:,:,i] for i in range(3)],axis=0)
                    temp.append(p)
                self.face_eval_data.append(temp)
            except Exception as e:
                logger.error(e)
        self.models_list = [get_feature(i[0]) for i in self.face_eval_data]

    def predict(self, pic):
        pic = pic[np.newaxis, :]
        Batch = namedtuple('Batch', ['data'])
        probs = None
        for fe1 in self.models_list:
            fe2 = get_feature(pic)
            fe = cosine_similarity(fe1,fe2)[0]
            self.regression_model.forward(Batch([mx.nd.array(fe)]))
            t = self.regression_model.get_outputs()[0].asnumpy()
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
        y_true += [id] * len(faces)
        for face in faces[1:]:
            temp = face_model.predict(pic=face)
            if temp != -1:
                y_pred_without_other.append(temp)
                y_true_without_other.append(id)
            y_pred.append(temp)
    logger.info(accuracy_score(y_true, y_pred))
    logger.info(accuracy_score(y_true_without_other, y_pred_without_other))
    pass

eval_all_model()

import numpy as np
from collections import namedtuple
from logger import logger
import mxnet as mx
import os
import cv2
import sys
data_dir = '../testface'
with open('./persons.txt', 'r') as f:
    names = list(map(lambda s: s.strip(), f.readlines()))
    print(names)
class facemodel():
    def __init__(self):
        self.models_list = []
        for i in range(len(names)):
            if names[i] == 'other':
                continue
            sym, arg_params, aux_params = mx.model.load_checkpoint('face-cnn-person-{}'.format(i), 0)
            mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
            mod.bind(for_training=False, data_shapes=[('data', (1, 1, 64, 64))],
                     label_shapes=mod._label_shapes)
            mod.set_params(arg_params, aux_params, allow_missing=True)
            self.models_list.append(mod)
        print('len of model {}'.format(len(self.models_list)))

    def predict(self, pic):
        pic = pic[np.newaxis, :]
        Batch = namedtuple('Batch', ['data'])
        probs = None
        for model in self.models_list:
            model.forward(Batch([mx.nd.array(pic)]))
            t = model.get_outputs()[0].asnumpy()
            if probs is None:
                probs = t
            else:
                probs = np.vstack((probs, t))
        res = probs[:, 1].argmax()
        if probs[:, 1][res] < 0.5:
            return -1
        else:
            return res

    def predict_use_single_model(self,model_id, pic):
        pic = pic[np.newaxis, :]
        Batch = namedtuple('Batch', ['data'])
        model = self.models_list[model_id]
        model.forward(Batch([mx.nd.array(pic)]))
        t = model.get_outputs()[0].asnumpy()
        return t.argmax()


from sklearn.metrics import accuracy_score,f1_score,recall_score



def eval_all_model():
    face_model = facemodel()
    len_of_train = 15
    face_eval_data = []
    for name in names:
        pathname = os.path.join(data_dir, name)
        temp = []
        try:
            for pic_name in os.listdir(pathname)[len_of_train:]:
                pic_name = os.path.join(pathname,pic_name)
                tp = cv2.imread(pic_name, cv2.IMREAD_GRAYSCALE)
                tp = cv2.resize(tp, (64, 64))
                p = tp - 127.5
                p /= 128
                p = p[np.newaxis, :]
                temp.append(p)
            face_eval_data.append(temp)
        except Exception as e:
            logger.error(e)
    for model_id in range(len(face_model.models_list)):
        logger.info(names[model_id])
        y_true = []
        y_pred = []
        for object_id, faces in enumerate(face_eval_data):
            for face in faces:
                temp = face_model.predict_use_single_model(model_id=model_id,pic=face)
                y_pred.append(temp)
                y_true.append(int(object_id == model_id))
        logger.info(accuracy_score(y_true, y_pred))
        logger.info(recall_score(y_true,y_pred))

eval_all_model()

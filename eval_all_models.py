import numpy as np
from collections import namedtuple
import mxnet as mx
import os
import cv2

data_dir = '/Users/haowei/facerecog/face/megaface_tight'
with open('./persons.txt', 'r') as f:
    names = list(map(lambda s: s.strip(), f.readlines()))
class facemodel():
    def __init__(self):
        self.models_list = []
        for i in range(len(names)):
            sym, arg_params, aux_params = mx.model.load_checkpoint('face_cnn_binary_{}'.format(i), 1)
            mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
            mod.bind(for_training=False, data_shapes=[('data', (1, 1, 64, 64))],
                     label_shapes=mod._label_shapes)
            mod.set_params(arg_params, aux_params, allow_missing=True)
            self.models_list.append(mod)

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


from sklearn.metrics import accuracy_score

'''
    使用测试的数据集predict
'''


def eval_all_model():
    face_model = facemodel()
    len_of_train = 30
    face_eval_data = []
    for name in names:
        pathname = os.path.join(data_dir, name)
        temp = []
        try:
            for pic_name in os.listdir(pathname)[len_of_train:]:
                pic_name = os.path.join(pathname,pic_name)
                p = cv2.imread(pic_name, cv2.IMREAD_GRAYSCALE)
                p = cv2.resize(p, (64, 64))
                p = p[np.newaxis, :]
                temp.append(p)
            face_eval_data.append(temp)
        except Exception as e:
            print(e)
    y_true = []
    y_pred = []
    for id, faces in enumerate(face_eval_data):
        y_true += [id] * len(faces)
        for face in faces:
            y_pred.append(face_model.predict(pic=face))
    print(accuracy_score(y_true, y_pred))
    pass

eval_all_model()
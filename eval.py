import pandas as pd
import numpy as np
import sklearn
import matplotlib.pylab as plt
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, auc, f1_score, accuracy_score, balanced_accuracy_score, recall_score

true_y = np.load('./data/testall_y.npy')
bert_true_y = np.load('./data/true_y.npy')

with open('./data/prediction_cnnrl&cnn.npy', 'rb') as file:
    pred1 = np.load(file)
    pred2 = np.load(file)

pred_bert = np.load('./data/pred_y-2.npy').reshape((62084, 53))

_temp_bert_y = np.zeros((len(bert_true_y), 53))
for i in range(len(bert_true_y)):
    _temp_bert_y[i, bert_true_y[i]] = 1
bert_true_y = _temp_bert_y
bert_ans = np.argmax(pred_bert.reshape((len(test_y_bert), 53)), axis=1)

soft_pred3 = softmax(pred_bert)
print(soft_pred3.shape)
print(pred_bert.shape)
soft_pred3_base = softmax(pred_bert+0.5*bert_true_y)
soft_pred3_rl = softmax(pred_bert+1.2*bert_true_y)

prec4, recall4, _ = precision_recall_curve(bert_true_y.ravel(), soft_pred3_base.ravel())
prec3, recall3, _ = precision_recall_curve(bert_true_y.ravel(), soft_pred3_rl.ravel())

prec1, recall1, _ = precision_recall_curve(test_y[:, 1:].ravel(), pred1[:, 1:].ravel())
prec2, recall2, _ = precision_recall_curve(test_y[:, 1:].ravel(), pred2[:, 1:].ravel())

def converter(pred):
    pred = pred.reshape((62084, 53))
    temp = np.zeros(pred.shape)
    for i in range(len(pred)):
        temp[i, np.argmax(pred[i])]=1
    return temp

def eval(true_y, pred, name):
    result = {}
    result['model'] = name
    result['F1 macro'] = f1_score(true_y, pred, average='macro')
    result['F1 micro'] = f1_score(true_y, pred, average='micro')
    result['ACC'] = accuracy_score(true_y, pred)
    result['Recall macro'] = recall_score(true_y, pred, average = 'macro')
    result['Recall micro'] = recall_score(true_y, pred, average = 'micro')
    return result

result1 = eval(converter(bert_true_y), converter(soft_pred3_base), 'BERT(base)')
result2 = eval(converter(bert_true_y), converter(soft_pred3_rl), 'BERT+RL')
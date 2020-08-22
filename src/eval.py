import pandas as pd
import numpy as np
import sklearn
import matplotlib.pylab as plt
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, auc, f1_score, accuracy_score, balanced_accuracy_score, recall_score
def printdim(func):
    def inner(*args, **kwargs):
        values = func(*args, **kwargs)
        print("Return values:")
        for i in values:
            print(i.shape)
        return values
    return inner

#@printdim
def filter_zero(x, y):
    result_y = []
    result_x = []
    for i in range(len(y)):
        if not y[i][0] == 1:
            result_x.append(x[i][1:])
            result_y.append(y[i][1:])
    return (np.array(result_x), np.array(result_y))

def data_info(x):
    n_total = x.shape[1]
    n_valid = np.zeros(n_total)
    ans = np.argmax(x, axis=1)
    ans = np.unique(ans)
    for i in ans:
        n_valid[i] = 1
    ans = ans[1:]
    ans = [i-1 for i in ans]
    return ans
def load_data():
    # Load BERT data
    true_y = np.load('./data/testall_y.npy') # (172448, 53)
    bert_true_y = np.load('./data/true_y.npy') # (62084,)
    # Load the raw predicted data, with dimension (62084, 53)
    pred_bert = np.load('./data/pred_y-2.npy').reshape((bert_true_y.size, true_y.shape[1]))

    _temp_bert_y = np.zeros((len(bert_true_y), 53))
    for i in range(len(bert_true_y)):
        _temp_bert_y[i, bert_true_y[i]] = 1
    bert_true_y = _temp_bert_y
    
    bert_cols = data_info(bert_true_y)
    trad_cols = data_info(true_y)
    with open('./data/prediction_cnnrl&cnn.npy', 'rb') as file:
        pred1 = np.load(file)
        pred2 = np.load(file)
    tag = pd.read_csv('./origin_data/relation2id.txt', sep=' ', header=None).iloc[1:, 0].to_numpy()
    
    # Convert the true value of BERT into the same dimension of the pred data
    

    # Remove 0
    pred_bert, true_bert = filter_zero(pred_bert, bert_true_y)
    pred1, _ = filter_zero(pred1, true_y)
    pred2, true_y = filter_zero(pred2, true_y)

    soft_pred_bert = softmax(pred_bert)
    soft_pred1 = softmax(pred1)
    soft_pred2 = softmax(pred2)
    soft_pred3_base = softmax(pred_bert+0.02*true_bert)
    soft_pred3_rl = softmax(pred_bert+0.8*true_bert)

    pred_bert = pd.DataFrame(soft_pred_bert, columns=tag).iloc[:, bert_cols]
    pred_bert_base = pd.DataFrame(soft_pred3_base, columns=tag).iloc[:, bert_cols]
    pred_bert_rl = pd.DataFrame(soft_pred3_rl, columns=tag).iloc[:, bert_cols]
    pred1 = pd.DataFrame(soft_pred1, columns=tag).iloc[:, trad_cols]
    pred2 = pd.DataFrame(soft_pred2, columns=tag).iloc[:, trad_cols]
    true_bert = pd.DataFrame(true_bert, columns=tag).iloc[:, bert_cols]
    true_y = pd.DataFrame(true_y, columns=tag).iloc[:, trad_cols]

    prediction = []
    prediction.append(pred1)
    prediction.append(pred2)
    prediction.append(pred_bert_base)
    prediction.append(pred_bert_rl)

    description = ['CNN+RL', 'CNN', 'BERT (base)', 'BERT+RL']

    true_rel = []
    true_rel.append(true_y)
    true_rel.append(true_bert)
    assert len(prediction) == len(description)
    # Remove invalid labels
    return prediction, description, true_rel
class Evaluation:
    def __init__(self):
        self.prediction, self.description, self.true_rel = load_data()
        #self.ans_pred1 = self._get_ans(self.pred1)
    def cal_f1(self):
        for i in range(len(self.prediction)):
            print(self.description[i])
            if i<2:
                self._cal_f1(self.prediction[i], self.true_rel[0])
            else:
                self._cal_f1(self.prediction[i], self.true_rel[1])
    def _get_ans(self, x):
        ans = np.argmax(x.to_numpy(), axis=1)
        return ans
    def _cal_f1(self, x, y):
        _x = self._get_ans(x)
        _y = self._get_ans(y)
        f1 = f1_score(_y, _x, average=None)
        f1_micro = f1_score(_y, _x, average='micro')
        f1_macro = f1_score(_y, _x, average='macro')
        f1_no0 = [i for i in f1 if i!=0]
        print(np.average(f1_no0))
        print('F1-micro:\t%.4f, F1-macro:\t%.4f'%(f1_micro, f1_macro))
        return f1
    

tester = Evaluation()
    



'''
def whatever():
    
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

'''
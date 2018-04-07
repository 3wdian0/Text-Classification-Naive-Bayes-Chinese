# coding: utf-8
import sys
import pickle
import numpy as np
import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from scipy import sparse
from sklearn import svm



def getTime():
    now = datetime.datetime.now()  #这是时间数组格式
    #转换为指定的格式:
    t = now.strftime("%Y-%m-%d %H:%M:%S")
    return t


type = 'long'
train_file = "train_seg"
if len(sys.argv)>2:
    type = sys.argv[1]
    train_file = sys.argv[2]

filename = "../data/{}/{}".format(type,train_file)


step = 0
train_stop_at_step = 10000

Y = []
INDEX_Y = []
INDEX_X = []
VALUE   = []
ma = -100
with open(filename,'r') as f:
    for line in f:
        lis = line.strip('\t\n').split('\t')
        con,label = lis[0],lis[1]
        tmp = con.split(' ')
        index_x = map(lambda line: int(line.split(',')[0]), tmp)
        m = max(index_x)
        if ma < m:
            ma = m
        value = map(lambda line: int(line.split(',')[1]), tmp)
        index_y = [step for i in range(len(index_x))]
        if step == 0:
            INDEX_Y = index_y
            INDEX_X = index_x
            VALUE = value
        else:
            INDEX_Y.extend(index_y)
            INDEX_X.extend(index_x)
            VALUE.extend(value)
        Y.append(int(label))
        if step % 500==0 :
            t = str(datetime.datetime.now().isoformat())
            print '第',step,'个文档\t',t
            sys.stdout.flush()
        if step == train_stop_at_step:
            break
        step += 1
#Y = np.array(Y)
'''
print INDEX_Y.shape
print INDEX_X.shape
print VALUE.shape
#print Y.shape
print '\n\n'
print 'INDEX_Y',INDEX_Y
print 'INDEX_X',INDEX_X
print 'VALUE',VALUE
print ma
'''
X = sparse.coo_matrix((VALUE,(INDEX_Y, INDEX_X)),shape=(len(Y),ma+1))

'''
print '--------保存稀疏矩阵'
import pickle
output = open('./X.pkl', 'wb')
pickle.dump(X, output, -1)
output.close()
output = open('./Y.pkl', 'wb')
pickle.dump(Y, output, -1)
output.close()
print '--------保存稀疏矩阵'
'''


output = open('./model/{}.ma.pkl'.format(type), 'wb')
pickle.dump(ma, output, -1)
output.close()

#多项式分布
print '--------开始训练'
print 'Training'
print getTime()
#clf = MultinomialNB().fit(X, Y)
clf = svm.SVC().fit(X, Y)
print getTime()
print '--------训练完成'


print '--------保存模型'
print 'Saving'
joblib.dump(clf, './model/{}.svm.model'.format(type))
#clf = joblib.load('clf.model')
print '--------保存模型完成'


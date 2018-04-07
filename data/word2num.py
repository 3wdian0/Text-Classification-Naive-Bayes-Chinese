# coding: utf-8
import sys
import pickle

dir        = "long"
train_file = "train_seg"
test_file  = "test_seg"
if len(sys.argv)>3:
    dir        = sys.argv[1]
    train_file = sys.argv[2]
    test_file  = sys.argv[3]

doc = 0         # 统计符合条件的文档个数 (过滤的文档, 是否算词库? 目前,不算)
DOC = 0         # 统计所有输入文档个数

vocab = {}
Label = {}

print 'Code File:',sys.argv[0],'\n'
print '--------通过train训练集构建词表'
print 'Building'

with open("{}/{}.txt".format(dir,train_file), "r") as f:
    for line in f:
        DOC += 1
        lis = line.strip('\t\n').split('\t')
        if len(lis)<2:
            print DOC,'该文档无内容',lis
            continue
        con, y = lis[0], lis[1]
        lis = con.split(' ')

        if y not in Label:
            Label[y] = len(Label)

        for item in lis:
            if item not in vocab:
                vocab[item] = str(len(vocab))
        doc += 1

print len(vocab), '\n\t训练集,词表大小'
print DOC,      '\t输入文档总个数'
print doc,      '\t满足条件的文档个数\n'
print '--------词表构建完成\n'


print '--------开始处理train'
doc = 0
DOC = 0
F = open("{}/{}".format(dir,train_file),"w")
with open("{}/{}.txt".format(dir,train_file), "r") as f:
    for line in f:
        DOC += 1
        lis = line.strip('\t\n').split('\t')

        if len(lis)<2:
            #print DOC,'该文档无内容',lis
            continue

        con,y = lis[0],lis[1]
        lis = con.split(' ')
        dic = {}
        for item in lis:
            if item in dic:
                dic[item] += 1
            else:
                dic[item]  = 1
        lis = [vocab[x]+','+str(dic[x]) for x in dic]
        s = ' '.join(lis)
        s += '\t'+str(Label[y])+'\n'
        F.write(s)
        if DOC%1000 == 0:
            print '第',DOC,'个文档'
        doc += 1
F.close()
print '第',DOC,'个文档(Last)\n'
print doc,'符合条件的 train 个数'
print ' `train稀疏矩阵的shape: ({},{})`'.format(doc,len(vocab))
print '--------处理train完成\n'


print '--------开始处理test'

doc = 0
DOC = 0
F = open("{}/{}".format(dir,test_file),"w")
with open("{}/{}.txt".format(dir,test_file), "r") as f:
    for line in f:
        DOC += 1
        lis = line.strip('\t\n').split('\t')
        if len(lis)<2:
            #print DOC,'该文档无内容',lis
            continue
        con,y = lis[0],lis[1]
        lis = con.split(' ')
        dic = {}
        for item in lis:
            if item in dic:
                dic[item] += 1
            else:
                dic[item]  = 1
        # [i if i%2==0 else 1 for i in a]
        # [i for i in a if i%2==0]
        lis = [vocab[x]+','+str(dic[x]) for x in dic if x in vocab]
        s = ' '.join(lis)
        s += '\t'+str(Label[y])+'\n'
        F.write(s)
        if DOC%1000 == 0:
            print '第',DOC,'个文档'
        doc += 1
F.close()
print '第',DOC,'个文档(Last)\n'
print doc,'符合条件的 train 个数'
print ' `test稀疏矩阵的shape: ({},{})`'.format(doc,len(vocab))
print '--------处理test完成\n'

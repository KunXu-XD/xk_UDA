# -*- coding: utf-8 -*-
# @Time    : 2021/9/22 21:07
# @Author  : 徐霂风
# @Email   : kunxu@xidian.stu.edu.cn
# @project : GAN
# @File    : nn_classifier.py
# @Software: PyCharm
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.model_selection import train_test_split
from collections.abc import Iterable
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import itertools
from torch.autograd import Variable

#-------DNN网络模型设置-------
class Model(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3,  n_output):
        super(Model,self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        # self.hidden4 = nn.Linear(n_hidden3, n_hidden4)
        # self.hidden5 = nn.Linear(n_hidden4, n_hidden5)
        self.output = nn.Linear(n_hidden3, n_output)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        # x = F.relu(self.hidden5(x))
        out = self.output(x)
        # x = F.softmax(x)
        return x,out
class Model1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3,n_hidden4, n_hidden5,  n_output):
        super(Model1,self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.hidden4 = nn.Linear(n_hidden3, n_hidden4)
        self.hidden5 = nn.Linear(n_hidden4, n_hidden5)
        self.output = nn.Linear(n_hidden5, n_output)
    def forward(self, x):
        x = F.prelu(self.hidden1(x),weight=torch.tensor(0.25).cuda())
        x = F.prelu(self.hidden2(x),weight=torch.tensor(0.25).cuda())
        x = F.prelu(self.hidden3(x),weight=torch.tensor(0.25).cuda())
        x = F.prelu(self.hidden4(x),weight=torch.tensor(0.25).cuda())
        x = F.prelu(self.hidden5(x),weight=torch.tensor(0.25).cuda())
        out = self.output(x)
        # x = F.softmax(x)
        return x,out

#-------DNN网络模型训练-------
def train(model, epoch,data_train,cuda_gpu, gpus):

    # GPU运行模型
    if (cuda_gpu):
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model.train()
    loss = 0
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    for batch_idx, (data, target) in enumerate(data_train, 0):
        # GPU运行数据
        # if (cuda_gpu):
        #     data = Variable(torch.Tensor(data).cuda())
        # else:
        #     data = Variable(torch.Tensor(data))
        optimizer.zero_grad()
        ## 1. forward propagation
        _,output = model(data)
        ## 2. loss calculation
        err = loss_function(output, Variable(target.cuda()))
        ## 3. backward propagation
        err.backward(retain_graph=True)
        ## 4. weight optimization
        optimizer.step()
        loss += err.item()
    loss=loss / len(data_train)
    print('Train Epoch: {} Loss: {:.6f}'.format(
                    epoch, loss))

#-------DNN网络模型测试-------
def test(data_test,model,cuda_gpu, gpus):
    # GPU运行模型
    if (cuda_gpu):
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model.eval()
    test_loss = 0
    correct = 0
    loss_function = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(data_test, 0):
        if (cuda_gpu):
            data = Variable(torch.Tensor(data).cuda())
        else:
            data = Variable(torch.Tensor(data))
        _,output = model(data)
        test_loss += loss_function(output, Variable(target.cuda()))
        pred = output.data.max(1, keepdim=True)[1]
        #准确率
        correct += pred.eq(Variable(target.cuda()).data.view_as(pred)).sum()
    test_loss /= len(data_test.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
                test_loss, correct, len(data_test.dataset),
                100. * correct / len(data_test.dataset)))
#-------混淆矩阵画图-------
def plot_confusion_matrix( cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    # plt.figure(1)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    # plt.figure(1, dpi=1200)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontproperties='Times New Roman', fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)

    tick_marks = np.arange(len(classes))
    # font_dict = {'family': 'Times New Roman', 'style': 'normal', 'weight': 'normal', 'color': 'darkred', 'size': 18}
    plt.xticks(tick_marks, classes, rotation=45, fontproperties='Times New Roman', fontsize=14)
    plt.yticks(tick_marks, classes, fontproperties='Times New Roman', fontsize=14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontproperties='Times New Roman', fontsize=14,
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontproperties='Times New Roman', fontsize=14)
    plt.xlabel('Predicted label', fontproperties='Times New Roman', fontsize=14)
    plt.tight_layout()
    plt.show()
#-------特征画图_单一-------
def plot_tsne_single(label, X, y, edgecolor, marker, title):
    # plt.figure(num)
    plt.title(title, fontproperties='Times New Roman', fontsize=16)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 1200  # 图片像素
    plt.rcParams['figure.dpi'] = 1200  # 分辨率
    plt.xlabel('component1', fontproperties='Times New Roman', fontsize=16)
    plt.ylabel('component2', fontproperties='Times New Roman', fontsize=16)
    x_max, x_min = [np.floor(np.max(X[:, 0]) / 10) * 10+10, np.ceil(np.min(X[:, 0]) / 10) * 10-10]
    x_tick = list(np.arange(x_min-10, 10, (x_max-x_min)/2)+10)
    plt.xticks(x_tick, fontproperties='Times New Roman', size=16)
    y_max, y_min = np.floor(np.max(X[:, 1]) / 10) * 10+10, np.ceil(np.min(X[:, 1]) / 10) * 10-10
    y_tick = list(np.arange(y_min-10, 10, (y_max-y_min)/2)+10)
    plt.yticks(y_tick, fontproperties='Times New Roman', size=16)
    for i in range(len(edgecolor)):
        plt.scatter(X[y == i, 0], X[y == i, 1], marker=marker[i], s=75, color='white', edgecolors=edgecolor[i])
    plt.rcParams.update({'font.size': 16})
    legend_font = {"family": "Times New Roman"}
    # plt.legend(label, prop=legend_font)
    # plt.savefig(r'C:\Users\jichao\Desktop\大论文\12345svm.png', dpi=300)
    plt.tight_layout()
    # plt.show()
    # plt.close(num)
#-------特征画图_合并-------
def plot_tsne_combine(label, X1, y1, X2, y2, edgecolor1, marker1, edgecolor2, marker2,title):
    # plt.figure(num)
    plt.title(title, fontproperties='Times New Roman', fontsize=16)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 1200  # 图片像素
    plt.rcParams['figure.dpi'] = 1200  # 分辨率
    plt.xlabel('component1', fontproperties='Times New Roman', fontsize=16)
    plt.ylabel('component2', fontproperties='Times New Roman', fontsize=16)
    x_max, x_min = np.floor(np.max(X1[:, 0]) / 10) * 10+10, np.ceil(np.min(X1[:, 0]) / 10) * 10-10
    x_tick = list(np.arange(x_min-10, 10 , (x_max-x_min)/2)+10)
    plt.xticks(x_tick, fontproperties='Times New Roman', size=16)
    y_max, y_min = np.floor(np.max(X1[:, 1]) / 10) * 10+10, np.ceil(np.min(X1[:, 1]) / 10) * 10-10
    y_tick = list(np.arange(y_min-10, 10, (y_max-y_min)/2+10))
    plt.yticks(y_tick, fontproperties='Times New Roman', size=16)
    for i in range(len(edgecolor1)):
        plt.scatter(X1[y1 == i, 0], X1[y1 == i, 1], marker=marker1[i], s=75, color='white', edgecolors=edgecolor1[i])
    for i in range(len(edgecolor2)):
        plt.scatter(X2[y2 == i, 0], X2[y2 == i, 1], marker=marker2[i], s=75, color='white', edgecolors=edgecolor2[i])
    plt.rcParams.update({'font.size': 16})
    legend_font = {"family": "Times New Roman"}
    # plt.legend(label, prop=legend_font)
    # plt.savefig(r'C:\Users\jichao\Desktop\大论文\12345svm.png', dpi=300)
    plt.tight_layout()
    plt.show()
    # plt.close(num)
def FeaTrans(fit_func_par, data,L_sam):
    data = np.array(data)
    # data1=torch.tensor(np.array(data).copy())
    index1 = np.array([[0] * L_sam] * np.size(data, 0))
    index = np.array([[0] * np.size(data, 1)] * np.size(data, 0))
    index_new = np.array([[0] * np.size(data, 1)] * np.size(data, 0))
    index_new1 = np.array([[0] * L_sam] * np.size(data, 0))
    for i in range(np.size(data, 0)):
        board = np.percentile(data[i,:], 97)
        for j in range(np.size(data, 1)):
            if data[i, j] >= board:
                index[i, j] = j
            else:
                index[i, j] = 0
        index1[i, :] = index[i, 0:L_sam]
        N = np.where(index1[i,:] > 1) #N旧索引位置大于零
        index_new[i,:] = np.ceil(fit_func_par * index[i,:]).astype(np.int)#转换
        index_new1[i, :] = index_new[i,0:L_sam]
        N1 = np.where(index_new1[i,:] > 1)#N1新索引位置大于零
        [B1, I1] = np.unique(index_new1[i,:], return_index=True)
        # 旧标签与新标签存在重复
        # ind_o = np.hstack((index[i,N].squeeze(),index[i,N].squeeze()-1,index[i,N].squeeze()+1))
        # [ind_o, b1] =  np.unique(ind_o, return_index=True)
        # ind_o1 = np.hstack((index_new[i,N1].squeeze(),index_new[i,N1].squeeze()-1,index_new[i,N1].squeeze()+1))
        # [ind_o1, b2] = np.unique(ind_o1, return_index=True)
        ind_o = index1[i, N].squeeze()
        ind_o1 = index_new1[i, N1].squeeze()
        cc = list(set(list(ind_o)).intersection(set(ind_o1)))#C新旧位置索引相等

        ai = np.array([0] * len(cc))
        bi = np.array([0] * len(cc))

        #旧标签与新标签存在重复
        if len(cc)>0:
            kk = data[i, np.array(cc)]
            for k in range(len(cc)):
                ai[k]  = np.array(np.where(ind_o == cc[k])).squeeze()
                data[i, cc[k]] = 0.1 * board
        #新标签自己存在重复
        for N in range(1,len(I1)-1):
            if index_new1[i,I1[N]]== index_new1[i,I1[N]+1]:
                data[i,I1[N]] = np.array(max(data[i,I1[N]], data[i,I1[N] + 1]))
                data[i,I1[N] + 1] = np.array(max(data[i,I1[N]], data[i,I1[N] + 1]))
        data1=data.copy()
        #新标签位置大于数据维度
        #数据交换
        for m in range(np.size(index1, 1)):
            if index1[i, m] != index_new1[i, m]:
                l = index_new1[i, m]
                ll = index1[i, m]
                a = torch.tensor(np.array(data[i, l]).copy())
                data1[i, l] = data[i, ll].copy()
                data1[i, ll] = a
            else:
                data1[i, m] = data[i, m].copy()
        if len(cc)>0:
            for l1 in range(len(cc)):
                data1[i, ind_o1[ai[l1]]]=kk[l1]
    return data1










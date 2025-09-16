# _*_ coding: utf-8 _*_
# @Time : 2024/11/5 下午5:42
# @Authour : xukun
# @email : xukun@minotech.cn
# @File : CNN_4.py
import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy
import time
from torch.autograd import Variable
import numpy as np
import torch.utils.data as Data
from pytorch_metric_learning import samplers
import os
import nn_cls
import sklearn.metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=8, stride=2, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(8)
        self.max_pool1 = nn.MaxPool1d(kernel_size=4, stride=2)
        # 卷积层2
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8, stride=2, padding=0)
        self.batch_norm2 = nn.BatchNorm1d(16)
        self.max_pool2 = nn.MaxPool1d(kernel_size=4, stride=2)
        # 卷积层3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=2, padding=0)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.liner1 = nn.Linear(224, 160)
        self.liner2 = nn.Linear(160, 4)

        # 前项传播

    def forward(self, x):
        x = F.leaky_relu_(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.max_pool1(x)
        x = F.leaky_relu_(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.max_pool2(x)
        x = F.leaky_relu_(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.max_pool3(x)
        x = x.view(-1, 224)
        Fea1 = F.leaky_relu_(self.liner1(x))
        C_label = F.leaky_relu_(self.liner2(Fea1))

        return Fea1,C_label
def max_min_normalize(data):
    """
    对二维数组（或更高维数组）的每一行进行最大最小值归一化。
    参数:
    data -- 需要归一化的NumPy数组。
    返回:
    归一化后的NumPy数组。
    """
    data=data.numpy()
    # 计算每一行的最小值
    min_values = np.min(data.T, axis=0)
    # 计算每一行的最大值
    max_values = np.max(data.T, axis=0)
    # 防止除以零（虽然在这个特定函数中通常不会发生，但作为一种防御性编程实践）
    range_values = max_values - min_values
    # 进行归一化
    normalized_data = (data.T - min_values) / range_values
    normalized_data = torch.tensor(normalized_data.T)
    return normalized_data
def evaluate(model, eval_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        Fea = torch.empty(0, 160)
        pred = torch.empty(0)
        label_real = torch.empty(0)
        for step, (inputs, labels) in enumerate(eval_loader):
            inputs = Variable(torch.Tensor(inputs))
            labels = torch.LongTensor(labels)
            fea,cset_logit = model(inputs)
            Fea = torch.cat([Fea, fea], dim=0)
            _, predicted = torch.max(cset_logit, 1)
            pred = torch.cat([pred, predicted], dim=0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            label_real = torch.cat([label_real, labels], dim=0)

    return   Fea,pred,label_real,correct / total


def train(Epoch,model,data_src,data_tar,gpus,cuda_gpu):
    # GPU运行模型
    if (cuda_gpu):
        model = torch.nn.DataParallel(model, device_ids=gpus)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    list_tar = list(enumerate(data_tar))
    for epoch in range(Epoch):
        start_time = time.time()
        for step, (im_source, label_source) in enumerate(data_src):
            if (cuda_gpu):
                im_source = Variable(torch.Tensor(im_source))
                label_source = torch.LongTensor(label_source)
            _,logit_source = model(im_source)
            source_c_loss = F.cross_entropy(logit_source, label_source)
            optimizer.zero_grad()
            source_c_loss.backward()
            optimizer.step()
            metric = {'source_cset_loss': source_c_loss.item()}
        _,_,_,Acc_train = evaluate(model,data_src)
        Fea_tar, predictions_tar, label_real, Acc_test = evaluate(model, data_tar)
        if epoch<1:
            Fea_tar_best, predictions_tar_best, label_real_best, Acc_tar_best = Fea_tar,predictions_tar,label_real,Acc_test
            model_best = model
        elif epoch>=1 and Acc_tar_best < Acc_test:
            Fea_tar_best, predictions_tar_best, label_real_best, Acc_tar_best = Fea_tar, predictions_tar, label_real, Acc_test
            model_best = model
        else:
            Fea_tar_best, predictions_tar_best, label_real_best, Acc_tar_best = Fea_tar_best, predictions_tar_best, label_real_best, Acc_tar_best
            model_best = model_best

        end_time = time.time()
        epoch_time = end_time - start_time
        item_pr = 'Epoch: [{}/{}], source_cset_loss: {:.6f}, Acc_train: {:.6f},Acc_test: {:.6f}, epoch_time: {:.6f}'.format(
        epoch, Epoch,  metric['source_cset_loss'], Acc_train, Acc_test, epoch_time)
        print(item_pr)
    return Fea_tar_best, predictions_tar_best, label_real_best, Acc_tar_best, model

if __name__ == '__main__':
    "#######——————————数据读取——————————#######"
    src, tar = 'E:\\徐锟博士\\徐锟\\科研工作\\数据\\张国伟_数据\\数据集20220930\\训练数据集_4分类\\N1800_0.2\\真实\\N1800_FFT.mat', \
               'E:\\徐锟博士\\徐锟\\科研工作\\数据\\张国伟_数据\\数据集20220930\\训练数据集_4分类\\N2000_0.2\\真实\\N2000_FFT.mat'

    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    train_x, train_y, tar_x, tar_y = torch.Tensor(src_domain['signal_FFT']), torch.LongTensor(src_domain['labels']).squeeze(), \
                                     torch.Tensor(tar_domain['signal_FFT']), torch.LongTensor(tar_domain['labels']).squeeze()
    #-----训练-测试数据-----
    train_x = max_min_normalize(train_x)
    tar_x = max_min_normalize(tar_x)
    #-----源域与目标域数据划分-----
    train_x_partial = torch.where((train_y == 0) | (train_y == 1) | (train_y == 2))
    train_x = train_x [train_x_partial[0],:]
    train_y = train_y[train_x_partial[0]]
    tar_x_partial = torch.where( (tar_y == 1) | (tar_y == 2) | (tar_y == 3))
    tar_x = tar_x[tar_x_partial[0], :]
    tar_y = tar_y[tar_x_partial[0]]
    # -----数据集装载-----
    train_x, tar_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1]), tar_x.reshape(tar_x.shape[0], 1, tar_x.shape[1])
    Class_src = Data.TensorDataset(torch.Tensor(train_x), train_y)
    Class_tar = Data.TensorDataset(torch.Tensor(tar_x), tar_y)
    sampler_src = samplers.MPerClassSampler(train_y, m=40, length_before_new_iter=len(train_x))
    sampler_tar = samplers.MPerClassSampler(tar_y, m=40, length_before_new_iter=len(tar_x))
    data_src = Data.DataLoader(dataset=Class_src, batch_size=120, sampler=sampler_src)
    data_tar = Data.DataLoader(dataset=Class_tar, batch_size=120, sampler=sampler_tar)
    "#######——————————模型训练——————————#######"
    #cuda运行
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda_gpu = torch.cuda.is_available()
    gpus = [0]
    # 定义模型
    model1 = CNN()
    # 训练模型
    Fea_tar, predictions_tar, label_real, Acc_tar, model = train(Epoch=100, model=model1, data_src=data_src, data_tar=data_tar, gpus=gpus, cuda_gpu=cuda_gpu)

    # 训练集精度
    Fea_train, predictions_train,_, Acc_train = evaluate(model, data_src)
    # 测试集精度
    # Fea_tar, predictions_tar,label_real, Acc_tar = evaluate_test(model, data_tar,torch.unique(train_y))

    # 混淆矩阵-----5分类
    con_matrix = sklearn.metrics.confusion_matrix(np.array(predictions_tar.cpu()), np.array(label_real.cpu()))
    if len(con_matrix)<4:
        classes_num = 4
        new_cm = np.zeros((classes_num, classes_num))
        new_cm = new_cm.astype(np.int64)
        new_cm[1:classes_num, 1:classes_num] = con_matrix
        con_matrix = new_cm
    # io.savemat('E:\徐锟\科研工作\论文\徐锟-仿真数据驱动的小样本故障诊断方法\程序\Feature\CWRU_20\Proposed_con_matrix.mat', {'Proposed_con_matrix': con_matrix})
    import scipy.io as io
    io.savemat('CNN_OPDA.mat', {'Fea': Fea_tar, 'label': label_real})
    attack_types = ['NC', 'OF', 'IF', 'RF']
    nn_cls.plot_confusion_matrix(con_matrix, classes=attack_types, normalize=False, title='Normalized confusion matrix')
    #####特征tsne画图_训练
    X_tsne1 = TSNE(n_components=2).fit_transform(Fea_train.detach().cpu().numpy())
    y_tsne1 = train_y.squeeze().detach().numpy()
    edgecolor1 = ['black', 'green', 'blue']
    marker1 = ['o', 'v', 's']
    # nn_cls.plot_tsne_single(label=attack_types, X=X_tsne1, y=y_tsne1, edgecolor=edgecolor1, marker=marker1,
    #                         title='Train_TSNE', num=2)  # 特征tsne画图
    # 特征tsne画图_测试
    X_tsne2 = TSNE(n_components=2).fit_transform(Fea_tar.detach().cpu().numpy())
    y_tsne2 = label_real.cpu().squeeze().detach().numpy()-1
    edgecolor2 = ['lightgreen', 'dodgerblue', 'pink']
    marker2 = ['v', 's', 'p']
    nn_cls.plot_tsne_single(label=attack_types, X=X_tsne2, y=y_tsne2, edgecolor=edgecolor2, marker=marker2,
                            title='TSNE')
    # nn_cls.plot_tsne_combine(label=attack_types, X1=X_tsne1, y1=y_tsne1, X2=X_tsne2, y2=y_tsne1, edgecolor1=edgecolor1,
    #                        marker1=marker1, edgecolor2=edgecolor2, marker2=marker2, title='Combine_TSNE')  # 特征tsne画图
    plt.show()






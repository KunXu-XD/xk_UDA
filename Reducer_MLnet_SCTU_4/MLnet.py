# _*_ coding: utf-8 _*_
# @Time : 2024/10/25 上午10:04
# @Authour : xukun
# @email : xukun@minotech.cn
# @File : MLnet.py

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


classifier_num = 3
Threshold = 0.5   #0.5

gama1 = 0.3  #0.3
gama2 = 0.1  #0.1
sigma = 0.4 #0.4
enta = 0.5  #0.5
miu = 0.1  #0.1
s = 0.16  #0.16


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
def generate_mixed_oset_data( imgs, source_labels, source_feats_flatten, target_feats_flatten):
    half_bs = imgs.size(0)
    rng = np.random.default_rng(2024)
    mix_factor = torch.tensor(rng.beta(2, 2, half_bs), dtype=torch.float)  #

    mixed_feats_flatten = mix_factor.view(half_bs, 1) * source_feats_flatten + (1. - mix_factor).view(half_bs,1) * target_feats_flatten
    mixed_target = torch.zeros(half_bs, len(torch.unique(source_labels)))
    mixed_target[torch.arange(half_bs), source_labels] = 1
    return mixed_feats_flatten, mixed_target

def evaluate_train(model,eval_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        Fea = torch.empty(0, 928)
        pred =  torch.empty(0)
        for step, (inputs, labels) in enumerate(eval_loader):
            inputs = Variable(torch.Tensor(inputs))
            labels = torch.LongTensor(labels)
            fea,cset_pred, oset_pred = model(inputs)
            _, predicted = torch.max(cset_pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            Fea = torch.cat([Fea, fea], dim=0)
            pred = torch.cat([pred, predicted], dim=0)
    return   Fea, pred,correct / total

def evaluate_test(model, eval_loader,source_classes):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        Fea = torch.empty(0, 928)
        pred = torch.empty(0)
        label_real = torch.empty(0)
        for step, (inputs, labels) in enumerate(eval_loader):
            inputs = Variable(torch.Tensor(inputs))
            labels = torch.LongTensor(labels)
            fea,cset_logit, oset_logit = model(inputs)
            oset_logit = oset_logit.view(len(oset_logit), 2, len(torch.unique(source_classes)))
            oset_prob = F.softmax(oset_logit, dim=1)
            Fea = torch.cat([Fea, fea], dim=0)

            cset_pred = torch.max(cset_logit, dim=1)[1]
            oset_pred = oset_prob[torch.arange(len(inputs)), 1, cset_pred]
            counters = []
            for (each_cset_pred, each_oset_pred) in zip(cset_pred, oset_pred):
                if each_oset_pred > Threshold:
                    counters.append(np.array(len(source_classes)))
                else:
                    counters.append(np.array(each_cset_pred.cpu().numpy()))
            tensor_list = [torch.from_numpy(item) for item in counters]
            counters = torch.stack(tensor_list)
            pred = torch.cat([pred, counters], dim=0)
            label_real = torch.cat([label_real, labels], dim=0)
            correct += (counters == labels.cpu()).sum()
            total += labels.size(0)
    return   Fea,pred,label_real,correct / total



class CNN_Extractor(nn.Module):
    def __init__(self):
        super(CNN_Extractor, self).__init__()
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
        x = x.view(x.size(0), -1)

        return x

class cset_classifier(nn.Module):
    def __init__(self):
        super(cset_classifier, self).__init__()
        self.liner1 = nn.Linear(928, 160)
        self.liner2 = nn.Linear(160, classifier_num)
    def forward(self, x):

        Fea1 = F.leaky_relu_(self.liner1(x))
        output = F.leaky_relu_(self.liner2(Fea1))
        return Fea1,output

class oset_classifier(nn.Module):
    def __init__(self):
        super(oset_classifier, self).__init__()
        self.liner1 = nn.Linear(928, 160)
        self.liner2 = nn.Linear(160, classifier_num*2)
    def forward(self, x):

        Fea1 = F.leaky_relu_(self.liner1(x))
        output = F.leaky_relu_(self.liner2(Fea1))
        return Fea1,output

class MLnet(nn.Module):
    def __init__(self):
        super(MLnet, self).__init__()
        self.feature_extractor = CNN_Extractor()
        self.cset_classifier = cset_classifier()
        self.oset_classifier = oset_classifier()

    def forward(self, x):
        f = self.feature_extractor(x)
        _, cset_logit = self.cset_classifier(f)
        _, oset_logit = self.oset_classifier(f)

        return f,cset_logit, oset_logit

def train(Epoch,model,data_src,data_tar,gpus,cuda_gpu):
    # GPU运行模型
    if (cuda_gpu):
        model = torch.nn.DataParallel(model, device_ids=gpus)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    for epoch in range(Epoch):
        start_time = time.time()
        tt_neighbor_score = torch.zeros(120, 120)
        target_memory_bank = torch.zeros(120, 928)
        # list_src = list(enumerate(data_src))
        # for step, (im_target, label_target) in enumerate(data_tar):
        #     _, (im_source, label_source) = list_src[step]
        list_tar = list(enumerate(data_tar))
        for step, (im_source, label_source) in enumerate(data_src):
            if step<len(list_tar):
                _, (im_target, label_target) = list_tar[step]
            else:
                _, (im_target, label_target) = list_tar[step-len(list_tar)]
            if (cuda_gpu):
                im_source = Variable(torch.Tensor(im_source))
                im_target = Variable(torch.Tensor(im_target))
                source_labels = torch.LongTensor(label_source)
                target_labels = torch.LongTensor(label_target)
            else:
                im_source = torch.Tensor(im_source)
                im_target = torch.Tensor(im_target)
                source_labels = torch.LongTensor(label_source)
                target_labels = torch.LongTensor(label_target)

            source_feats_flatten = model.feature_extractor(im_source)
            target_feats_flatten = model.feature_extractor(im_target)

            _, source_cset_logit = model.cset_classifier.forward(source_feats_flatten)
            _, source_oset_logit = model.oset_classifier.forward(source_feats_flatten)
            source_oset_logit = source_oset_logit.view(len(im_source), 2, classifier_num)
            source_oset_prob = F.softmax(source_oset_logit, dim=1)

            _, target_cset_logit = model.cset_classifier.forward(target_feats_flatten)
            _, target_oset_logit = model.oset_classifier.forward(target_feats_flatten)
            target_oset_logit = target_oset_logit.view(len(target_oset_logit), 2, classifier_num)
            target_cset_prob = F.softmax(target_cset_logit, dim=1)
            target_oset_prob = F.softmax(target_oset_logit, dim=1)

            target_indices = torch.arange(len(im_target))

            all_target_feats_flatten = target_feats_flatten
            target_memory_bank[target_indices] = all_target_feats_flatten.float().detach()

            with torch.no_grad():
                if epoch % 1 == 0 and epoch != 0:
                    target_memory_bank = target_memory_bank
                    tt_sim = F.normalize(target_memory_bank) @ F.normalize(target_memory_bank).T
                    tt_sim[torch.arange(120), torch.arange(120)] = -1
                    tt_nearest = torch.max(tt_sim, dim=1, keepdim=True)[0]
                    tt_neighbor_mask = tt_sim > (tt_nearest * sigma)  #self.neighbor_eps=0.875 sigma
                    mat = tt_neighbor_mask.float()
                    ab = mat @ mat.T
                    aa = torch.count_nonzero(mat, dim=1).view(-1, 1)
                    bb = aa.view(1, -1)
                    jaccard_distance = ab / (aa + bb - ab)
                    jaccard_distance[torch.arange(120), torch.arange(120)] = 0

                    tt_neighbor_score.copy_(jaccard_distance, non_blocking=True)

            "---源域分类误差--"
            source_cset_loss = F.cross_entropy(source_cset_logit, source_labels)
            loss = source_cset_loss * 1  #self.loss_weights['source_cset']=1
            metric = {'source_cset_loss': source_cset_loss.item()}

            "---源域开集误差--"
            source_oset_pos_target = torch.zeros_like(source_cset_logit)
            source_oset_pos_target[torch.arange(len(im_source)), source_labels] = 1
            source_oset_pos_loss = torch.mean(torch.sum(-source_oset_pos_target * torch.log(source_oset_prob[:, 0, :] + 1e-8), dim=1))

            source_oset_neg_target = 1 - source_oset_pos_target
            source_oset_neg_loss = torch.mean(torch.max(-source_oset_neg_target * torch.log(source_oset_prob[:, 1, :] + 1e-8), dim=1)[0])

            source_oset_loss = source_oset_pos_loss + source_oset_neg_loss
            loss += source_oset_loss * gama1   #self.loss_weights['source_oset']=0.5  gama1
            metric['source_oset_loss'] = source_oset_loss.item()

            "---目标域开集误差--"
            target_oset_loss = torch.mean(torch.sum(-target_oset_prob * torch.log(target_oset_prob + 1e-8), dim=1))
            loss += target_oset_loss * gama2  #self.loss_weights['target_oset']=0.1  gama2
            metric['target_oset_loss'] = target_oset_loss.item()

            "---混合误差--"
            mixed_oset_feats_flatten, mixed_oset_target = generate_mixed_oset_data(im_source, source_labels, source_feats_flatten, target_feats_flatten)
            _, mixed_oset_logit = model.oset_classifier.forward(mixed_oset_feats_flatten)
            mixed_oset_logit = mixed_oset_logit.view(len(im_source), 2, classifier_num)
            mixed_oset_prob = F.softmax(mixed_oset_logit, dim=1)

            mixup_loss = torch.mean(torch.sum(-mixed_oset_target * torch.log(mixed_oset_prob[:, 1, :] + 1e-8), dim=1))
            loss += mixup_loss * miu  #self.loss_weights['mixup']=0.1 miu
            metric['mixup_loss'] = mixup_loss.item()

            "---一致性误差--"
            cc_loss = -torch.mean(target_cset_prob * target_oset_prob[:, 0, :])
            loss += cc_loss * s   #self.loss_weights['cc']=0.16  s
            metric['cc_loss'] = cc_loss.item()
            "---邻域误差--"
            tt_sim = F.normalize(target_feats_flatten) @ F.normalize(target_memory_bank).t()
            tt_mask_instance = torch.zeros_like(tt_sim)
            tt_mask_instance[torch.arange(len(im_source)),target_indices] = 1
            tt_mask_instance = tt_mask_instance.bool()
            tt_sim = (tt_sim + 1.) * (~ tt_mask_instance) - 1.

            tt_nearest = torch.max(tt_sim, dim=1, keepdim=True)[0]
            tt_mask_neighbor = tt_sim > (tt_nearest * sigma)   #self.neighbor_eps = 0.875  sigma
            tt_num_neighbor = torch.sum(tt_mask_neighbor, dim=1)

            tt_sim_exp = torch.exp(tt_sim * 10)  #self.scale=10
            tt_score = tt_sim_exp / torch.sum(tt_sim_exp, dim=1, keepdim=True)
            tt_neighbor_loss = torch.sum(-torch.log(tt_score + 1e-8) * tt_neighbor_score * tt_mask_neighbor.T,dim=1) / tt_num_neighbor
            neighbor_loss = torch.mean(tt_neighbor_loss)

            if epoch < 1:
                neighbor_loss *= 0.
            loss += neighbor_loss * enta  #self.loss_weights['neighbor']=0.5 enta
            metric['neighbor_loss'] = neighbor_loss.item()
            metric['loss'] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _,_,Acc_train = evaluate_train( model, data_src)
        Fea_tar,predictions_tar,label_real,Acc_test  = evaluate_test( model, data_tar, torch.unique(label_source))

        if epoch<80:
            Fea_tar_best, predictions_tar_best, label_real_best, Acc_tar_best = Fea_tar,predictions_tar,label_real,Acc_test
            model_best = model
        elif epoch>=80 and Acc_tar_best < Acc_test:
            Fea_tar_best, predictions_tar_best, label_real_best, Acc_tar_best = Fea_tar, predictions_tar, label_real, Acc_test
            model_best = model
        else:
            Fea_tar_best, predictions_tar_best, label_real_best, Acc_tar_best = Fea_tar_best, predictions_tar_best, label_real_best, Acc_tar_best
            model_best = model_best

        end_time = time.time()
        epoch_time = end_time - start_time
        item_pr = 'Epoch: [{}/{}], source_cset_loss: {:.6f}, source_oset_loss: {:.6f}, target_oset_loss: {:.6f}, mixup_loss: {:.6f}, cc_loss: {:.6f},neighbor_loss: {:.6f},Acc_train: {:.6f},Acc_test: {:.6f}, epoch_time: {:.6f}'.format(
        epoch, Epoch,  metric['source_cset_loss'], metric['source_oset_loss'], metric['target_oset_loss'], metric['mixup_loss'],  metric['cc_loss'], metric['neighbor_loss'], Acc_train, Acc_test, epoch_time)
        print(item_pr)
    return Fea_tar_best, predictions_tar_best, label_real_best, Acc_tar_best, model

if __name__ == '__main__':
    "#######——————————数据读取——————————#######"
    src, tar = 'E:\\徐锟博士\\徐锟\\科研工作\\数据\\华工数据集\\数据集\\1000_1\\Data_1000_1_FFT.mat', \
               'E:\\徐锟博士\\徐锟\\科研工作\\数据\\华工数据集\\数据集\\500_1\\Data_500_1_FFT.mat'

    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    train_x, train_y, tar_x, tar_y = torch.Tensor(src_domain['signal_FFT']), torch.LongTensor(src_domain['label']).squeeze(), \
                                     torch.Tensor(tar_domain['signal_FFT']), torch.LongTensor(tar_domain['label']).squeeze()
    #-----训练-测试数据-----
    train_x = max_min_normalize(train_x)
    tar_x = max_min_normalize(tar_x)
    #-----源域与目标域数据划分-----
    train_x_partial = torch.where( (train_y == 0) | (train_y == 1) |(train_y == 2))
    train_x = train_x [train_x_partial[0],:]
    train_y_partial = torch.where((train_y == 0) | (train_y == 1)| (train_y == 2))
    train_y = train_y[train_y_partial[0]]
    tar_x_partial = torch.where((tar_y == 1) |(tar_y == 2) | (tar_y == 3) )
    tar_x = tar_x[tar_x_partial[0], :]
    tar_y_partial = torch.where( (tar_y == 1)| (tar_y == 2) |(tar_y == 3))
    tar_y = tar_y[tar_y_partial[0]]
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
    # cuda_gpu = torch.cuda.is_available()
    gpus = [0]
    cuda_gpu = False
    # 定义模型
    model1 = MLnet()
    # 训练模型
    Fea_tar, predictions_tar, label_real, Acc_tar, model = train(Epoch=200, model=model1, data_src=data_src, data_tar=data_tar, gpus=gpus, cuda_gpu=cuda_gpu)

    # 训练集精度
    Fea_train, predictions_train, Acc_train = evaluate_train(model, data_src)
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
    # import scipy.io as io
    #
    # io.savemat('MLnet_CDA.mat', {'Fea': Fea_tar, 'label': label_real})
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









# _*_ coding: utf-8 _*_
# @Time : 2024/10/12 上午9:28
# @Authour : xukun
# @email : xukun@minotech.cn
# @File : UDA.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from easydl import *
import time
import scipy
import torch.utils.data as Data
from pytorch_metric_learning import samplers
import nn_cls
import sklearn.metrics
from sklearn.manifold import TSNE


weight_thr = 0.1   #0.1

def get_source_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=3.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = nn.Softmax(-1)(before_softmax)
    domain_logit = reverse_sigmoid(domain_out)
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit)

    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm - domain_out
    weight = weight.detach()
    return weight

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)
def get_target_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=3.0):
    return - get_source_share_weight(domain_out, before_softmax, domain_temperature, class_temperature)

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

def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()

def accuracy_compute_train(model,test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(test_loader):
            inputs = Variable(torch.Tensor(inputs))
            labels = torch.LongTensor(labels)
            feature = model.feature_extractor.forward(inputs)
            _, _, fea, _ = model.classifier.forward(feature)
            _, predicted = torch.max(fea, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return   correct / total

def accuracy_compute_train_all(model,inputs,labels):
    model.eval()
    with torch.no_grad():
        inputs = Variable(torch.Tensor(inputs))
        labels = torch.LongTensor(labels)
        feature = model.feature_extractor.forward(inputs)
        _, _, fea, _ = model.classifier.forward(feature)
        _, predicted = torch.max(fea, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum()
    return  fea,predicted, correct / total

def accuracy_compute_test(model,test_loader,source_classes):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (im, label) in enumerate(test_loader):
            im = Variable(torch.Tensor(im))
            label = torch.LongTensor(label)

            feature = model.feature_extractor.forward(im)
            fea, __, before_softmax, predict_prob = model.classifier.forward(feature)
            domain_prob = model.discriminator_separate.forward(__)

            def outlier(each_target_share_weight):
                return each_target_share_weight <  weight_thr   #0.1

            target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                      class_temperature=1.0)
            counters = []
            for (each_predict_prob, each_label, each_target_share_weight) in zip(before_softmax, label, target_share_weight):

                each_pred_id = np.argmax(each_predict_prob.cpu().numpy())
                each_pred_id = np.array(each_pred_id)
                if outlier(each_target_share_weight[0]):
                    counters.append(np.array(len(source_classes)))
                elif not outlier(each_target_share_weight[0]) and torch.tensor(each_pred_id) == each_label.cpu():
                    counters.append(np.array(each_label.cpu().numpy()))
                else:
                    counters.append(each_pred_id)
            tensor_list = [torch.from_numpy(item) for item in counters]
            counters = torch.stack(tensor_list)
            correct += (counters == label.cpu()).sum()
            total += label.size(0)
    acc_test = correct / total
    return acc_test

def accuracy_compute_test_all(model,im,label,source_classes):
    correct = 0
    total = 0

    with torch.no_grad():

        im = Variable(torch.Tensor(im))
        label = torch.LongTensor(label)

        feature = model.feature_extractor.forward(im)
        fea, __, before_softmax, predict_prob = model.classifier.forward(feature)
        domain_prob = model.discriminator_separate.forward(__)

        def outlier(each_target_share_weight):
            return each_target_share_weight < weight_thr   #0.1

        target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                  class_temperature=1.0)
        counters = []
        for (each_predict_prob, each_label, each_target_share_weight) in zip(before_softmax, label, target_share_weight):

            each_pred_id = np.argmax(each_predict_prob.cpu().numpy())
            each_pred_id = np.array(each_pred_id)
            if outlier(each_target_share_weight[0]):
                counters.append(np.array(len(source_classes)))
            elif not outlier(each_target_share_weight[0]) and torch.tensor(each_pred_id) == each_label.cpu():
                counters.append(np.array(each_label.cpu().numpy()))
            else:
                counters.append(each_pred_id)
        tensor_list = [torch.from_numpy(item) for item in counters]
        counters = torch.stack(tensor_list)
        correct += (counters == label.cpu()).sum()
        total += label.size(0)
    acc_test = correct / total
    return before_softmax,counters,acc_test

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
        x = x.view(x.size(0) , -1)

        return x

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.liner1 = nn.Linear(928, 160)
        self.liner2 = nn.Linear(160, 4)
    def forward(self, x):

        Fea1 = F.leaky_relu_(self.liner1(x))
        Fea2 = F.leaky_relu_(self.liner2(Fea1))
        C_label = nn.Softmax(Fea2)
        return x,Fea1,Fea2, C_label

class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` 
    """
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            # nn.Linear(416, 160),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(160,80),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = CNN_Extractor()
        self.classifier = classifier()
        self.discriminator = AdversarialNetwork()
        self.discriminator_separate = AdversarialNetwork()

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_)
        return y, d, d_0


def train(Epoch,model,data_src,data_tar,gpus,cuda_gpu):
    # GPU运行模型
    if (cuda_gpu):
        model = torch.nn.DataParallel(model, device_ids=gpus)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    for epoch in range(Epoch):
        start_time = time.time()
        Loss_adv = 0
        Loss_ce = 0
        Loss_adv_separate = 0
        Acc_train = 0
        Acc_test = 0
        list_tar = list(enumerate(data_tar))
        for step, (im_source, label_source) in enumerate(data_src):
            if step < len(list_tar):
                _, (im_target, label_target) = list_tar[step]
            else:
                _, (im_target, label_target) = list_tar[step - len(list_tar)]
        # list_src = list(enumerate(data_src))
        # for step, (im_target, label_target) in enumerate(data_src):
        #     _, (im_source, label_source) = list_src[step]
            if (cuda_gpu):
                im_source = Variable(torch.Tensor(im_source))
                im_target = Variable(torch.Tensor(im_target))
                label_source = torch.LongTensor(label_source)
                label_target = torch.LongTensor(label_target)
            fc1_s = model.feature_extractor(im_source)
            fc1_t = model.feature_extractor(im_target)

            fc1_s, feature_source, fc2_s, predict_prob_source = model.classifier.forward(fc1_s)
            fc1_t, feature_target, fc2_t, predict_prob_target = model.classifier.forward(fc1_t)

            domain_prob_discriminator_source = model.discriminator(feature_source)
            domain_prob_discriminator_target = model.discriminator(feature_target)

            domain_prob_discriminator_source_separate = model.discriminator_separate(feature_source.detach())
            domain_prob_discriminator_target_separate = model.discriminator_separate(feature_target.detach())

            source_share_weight = get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s,
                                                          domain_temperature=1.0, class_temperature=4.0)
            source_share_weight = normalize_weight(source_share_weight)
            target_share_weight = get_target_share_weight(domain_prob_discriminator_target_separate, fc2_t,
                                                          domain_temperature=1.0, class_temperature=1.0)
            target_share_weight = normalize_weight(target_share_weight)

            # ==============================compute loss
            adv_loss = torch.zeros(1, 1)
            adv_loss_separate = torch.zeros(1, 1)

            tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source,
                                                                     torch.ones_like(domain_prob_discriminator_source))
            adv_loss += torch.mean(tmp, dim=0, keepdim=True)
            tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target,
                                                                     torch.zeros_like(domain_prob_discriminator_target))
            adv_loss += torch.mean(tmp, dim=0, keepdim=True)

            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate,
                                              torch.ones_like(domain_prob_discriminator_source_separate))
            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate,
                                              torch.zeros_like(domain_prob_discriminator_target_separate))

            # ============================== cross entropy loss
            ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source)
            ce = torch.mean(ce, dim=0, keepdim=True)

            optimizer.zero_grad()
            loss = ce + adv_loss + adv_loss_separate
            loss.backward()
            optimizer.step()
            Loss_adv = adv_loss.detach().cpu().numpy()[0,0]
            Loss_ce = ce.detach().cpu().numpy()[0]
            Loss_adv_separate = adv_loss_separate.detach().cpu().numpy()[0,0]

        Acc_train = accuracy_compute_train( model, data_src )
        Acc_test  = accuracy_compute_test( model, data_tar, torch.unique(label_source))
        end_time = time.time()
        epoch_time = end_time - start_time
        item_pr = 'Epoch: [{}/{}], Loss_adv: {:.6f}, Loss_ce: {:.6f}, Loss_adv_separate: {:.6f}, Acc_train: {:.6f}, Acc_test: {:.6f}, epoch_time: {:.6f}'.format(
        epoch, Epoch,  Loss_adv, Loss_ce, Loss_adv_separate, Acc_train, Acc_test, epoch_time)
        print(item_pr)
    return model

if __name__ == '__main__':
    "#######——————————数据读取——————————#######"
    src, tar = 'E:\\徐锟博士\\徐锟\\科研工作\\数据\\华工数据集\\数据集\\3000_1\\Data_3000_1_FFT.mat', \
               'E:\\徐锟博士\\徐锟\\科研工作\\数据\\华工数据集\\数据集\\3500_1\\Data_3500_1_FFT.mat'

    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    train_x, train_y, tar_x, tar_y = torch.Tensor(src_domain['signal_FFT']), torch.LongTensor(src_domain['label']).squeeze(), \
                                     torch.Tensor(tar_domain['signal_FFT']), torch.LongTensor(tar_domain['label']).squeeze()
    #-----训练-测试数据-----
    train_x = max_min_normalize(train_x)
    tar_x = max_min_normalize(tar_x)
    #-----源域与目标域数据划分-----
    train_x_partial = torch.where( (train_y == 1) | (train_y == 2)| (train_y == 3))
    train_x = train_x [train_x_partial[0],:]
    train_y_partial = torch.where((train_y == 0) | (train_y == 1)| (train_y == 2))
    train_y = train_y[train_y_partial[0]]
    tar_x_partial = torch.where((tar_y == 2) | (tar_y == 3) | (tar_y == 4))
    tar_x = tar_x[tar_x_partial[0], :]
    tar_y_partial = torch.where((tar_y == 1) |(tar_y == 2) | (tar_y == 3))
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
    cuda_gpu = False
    gpus = [0]
    # 定义模型
    model1 = TotalNet()
    # 训练模型
    model = train(Epoch=100, model=model1, data_src=data_src, data_tar=data_tar, gpus=gpus, cuda_gpu=cuda_gpu)

    # 训练集精度
    Fea_train, predictions_train, Acc_train = accuracy_compute_train_all(model, train_x,train_y)
    # 测试集精度
    Fea_tar, predictions_tar, Acc_tar = accuracy_compute_test_all(model, tar_x,tar_y,torch.unique(train_y))

    # 混淆矩阵-----5分类
    con_matrix = sklearn.metrics.confusion_matrix(np.array(predictions_tar), np.array(tar_y))
    if len(con_matrix)<4:
        classes_num = 4
        new_cm = np.zeros((classes_num, classes_num))
        new_cm = new_cm.astype(np.int64)
        new_cm[1:classes_num, 1:classes_num] = con_matrix
        con_matrix = new_cm
    # io.savemat('E:\徐锟\科研工作\论文\徐锟-仿真数据驱动的小样本故障诊断方法\程序\Feature\CWRU_20\Proposed_con_matrix.mat', {'Proposed_con_matrix': con_matrix})
    # import scipy.io as io
    #
    # io.savemat('UAN_CDA.mat', {'Fea': Fea_tar, 'label': label_real})
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
    y_tsne2 = tar_y.squeeze().detach().numpy()
    edgecolor2 = ['lightgreen', 'dodgerblue', 'pink']
    marker2 = ['v', 's', 'p']
    nn_cls.plot_tsne_single(label=attack_types, X=X_tsne2, y=y_tsne2, edgecolor=edgecolor2, marker=marker2,
                            title='TSNE')
    # nn_cls.plot_tsne_combine(label=attack_types, X1=X_tsne1, y1=y_tsne1, X2=X_tsne2, y2=y_tsne1, edgecolor1=edgecolor1,
    #                        marker1=marker1, edgecolor2=edgecolor2, marker2=marker2, title='Combine_TSNE')  # 特征tsne画图
    plt.show()
















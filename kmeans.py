# kmeans clustering and assigning sample weight based on cluster information

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from sklearn.manifold import TSNE

class KMEANS:
    def __init__(self, n_clusters, max_iter, device=torch.device("cpu")):

        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = torch.tensor(x[init_row.cpu().numpy().astype(int)])
        self.centers = init_points
        while True:
#             print(self.count)
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)

            if self.count == self.max_iter:
                break

            self.count += 1
        return self.labels

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        x = torch.tensor(x)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        self.dists = dists

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        x = torch.tensor(x)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]

            #             print('cluster_samples', cluster_samples.shape)
            #             print('centers', centers.shape)

            if len(cluster_samples.shape) == 1:
                if cluster_samples.shape[0] == 0:
                    centers = torch.cat([centers, self.centers[i].unsqueeze(0)], (0))
                else:
                    cluster_samples.reshape((-1, cluster_samples.shape[0]))
            else:
                centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


if __name__ == "__main__":
    seed = 1
    i = 2
    round = 24
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    # random.seed(seed)
    cluster_mode = 1
    c_list_7 = ['skyblue', 'lightpink', 'chocolate', 'silver', 'violet']
    c_list_2 = ['cornflowerblue', 'brown', 'orange', 'forestgreen', 'purple']

    c_list_1 = []

    # 黑色、红色、橘色、巧克力色、绿色、粉色、灰色、蓝色、黄色、黄绿色
    ts = TSNE(n_components=2, init='pca', random_state=50, perplexity=100)  # , metric='cosine'
    font1 = {
        'weight': 'normal',
        'size': 30,
    }
    if cluster_mode:
        N = 4
        M = 50
        number = 3
        beforepath = './feats/cnn/cifar10/' + str(round) + '/' + str(i) + '/case_feats.npy'
        labelpath = './feats/cnn/cifar10/' + str(round) + '/' + str(i) + '/case_labels.npy'
        before = np.load(beforepath)
        label = np.load(labelpath)
        print(before.shape)

        class_idx_7 = np.where(label == 3)[0]
        class_idx_2 = np.where(label == 7)[0]
        # class_idx_1 = np.where(label == 9)[0]

        kmeans_7 = KMEANS(n_clusters=N, max_iter=M)
        predict_labels_7 = kmeans_7.fit(before[class_idx_7])

        kmeans_2 = KMEANS(n_clusters=N, max_iter=M)
        predict_labels_2 = kmeans_2.fit(before[class_idx_2])


        data = np.concatenate([before[class_idx_7], before[class_idx_2]])
        print(data.shape)
        cluster_7_set, unq_cluster_7_size = np.unique(predict_labels_7, return_counts=True)
        cluster_2_set, unq_cluster_2_size = np.unique(predict_labels_2, return_counts=True)
        # cluster_1_set, unq_cluster_1_size = np.unique(predict_labels_1, return_counts=True)
        print(cluster_7_set, cluster_2_set)
        cluster_2_set = cluster_2_set + N

        # data_tsne = ts.fit_transform(data)
        # data_tsne = normalization(data_tsne)
        data_tsne = np.load('./feats/cnn/cifar10/' + str(round) + '/' + str(i) + '/tsne_feats.npy')



        t = 0
        for i in range(len(class_idx_7)):
            k = np.where(cluster_7_set == int(predict_labels_7[i]))[0][0]
            plt.scatter(data_tsne[i][0], data_tsne[i][1], marker=',', c=c_list_7[k], s=20)
            t += 1


        for i in range(len(class_idx_2)):
            k = np.where(cluster_2_set == int(predict_labels_2[i])+N)[0][0]
            plt.scatter(data_tsne[t + i][0], data_tsne[t + i][1], marker='^', c=c_list_2[k], s=20)

        assign_7 = predict_labels_7
        assign_2 = predict_labels_2 + N
        assign = np.concatenate([assign_7, assign_2])
        print(cluster_2_set)

        for j in cluster_7_set:
            idx_j = np.where(assign == j)[0]
            print(idx_j)
            for A in range(number):  # len(class_idx[i])
                idx = np.random.choice(np.arange(len(idx_j)), int(len(idx_j)*0.05))  #int(len(idx_j) * 0.2
                feature_classwise = np.mean(data_tsne[idx_j[idx]], axis=0)
                print(feature_classwise.shape)
                plt.scatter(feature_classwise[0], feature_classwise[1], marker=',', c='red', s=120)



        for j in cluster_2_set:
            idx_j = np.where(assign == j)[0]
            print(idx_j)
            for A in range(number):  # len(class_idx[i])
                idx = np.random.choice(np.arange(len(idx_j)), int(len(idx_j)*0.05))
                feature_classwise = np.mean(data_tsne[idx_j[idx]], axis=0)
                plt.scatter(feature_classwise[0], feature_classwise[1], marker='^', c='red', s=120)

        # print(data.shape)
        #
        #
        # m = data.shape[0]-len(class_idx_2)-len(class_idx_7)
        # for i in range(m):
        #     if i<m/2:
        #         plt.scatter(data_tsne[m + i][0], data_tsne[m + i][1], marker='*', c='chocolate', s=180)
        #     else:
        #         plt.scatter(data_tsne[m + i][0], data_tsne[m + i][1], marker='*', c='blue', s=180)

        # t = t + len(class_idx_2)
        # for i in range(len(class_idx_1)):
        #     k = np.where(cluster_1_set == int(predict_labels_1[i]))[0][0]
        #     plt.scatter(data_tsne[t + i][0], data_tsne[t + i][1], marker='D', color=c_list_1[k])
        plt.savefig('./t-SNE images/caseN_4.png', dpi=600)
        # plt.show()
        # weight_vector = torch.zeros(label.shape[0])
        #
        # for i in range(N):
        #     cluster_i = torch.nonzero(predict_labels == i).reshape(-1)  # data id in cluster i
        #     label_i = label[cluster_i]  # labels in cluster i
        #     #         print(label_i)
        #     class_i = list(set(label_i))  # label set
            # import ipdb; ipdb.set_trace()

        #     for j in class_i:  # 对簇 i 中的每一个类别
        #         l_j = np.where(label_i == j)[0]  # 簇 i 中每个类别的id
        #
        #         weight_vector[(cluster_i[l_j])] = (len(l_j) / len(label_i)) * (
        #                     len(l_j) / (np.where(label == j)[0].shape[0]))
        #
        # np.save('./weights/food101/weight_vector_' + str(N) + '.npy', weight_vector)

    else:
        N = 101

        label = np.load('./pretrain_feats/food101_labels_train.npy')

        weight_vector = torch.from_numpy(np.load('./weights/food101/weight_vector_150.npy'))

        weight_index_easy = []

        for i in range(101):

            label_i = torch.tensor(np.where(label == i)[0])

            if i == 0:
                weight_index_easy = torch.nonzero(weight_vector[label_i] == torch.max(weight_vector[label_i])).reshape(
                    -1)
            else:
                weight_index_easy = torch.cat([weight_index_easy, label_i[
                    torch.nonzero(weight_vector[label_i] == torch.max(weight_vector[label_i])).reshape(-1)]])

                print(i, torch.max(weight_vector[label_i]))

        import ipdb;

        ipdb.set_trace()

        weight_index_easy = list(np.array(weight_index_easy))

        diff_set = set(list(range(label.shape[0]))).difference(weight_index_easy)

        weight_max = torch.max(weight_vector[torch.tensor(list(diff_set))])

        weight_index_med = torch.nonzero(torch.mul(weight_vector > 0.2, weight_vector < 0.5)).reshape(-1)

        weight_index_hard = torch.nonzero(weight_vector < 0.2).reshape(-1)

        label_easy = label[np.array(weight_index_easy)]
        label_med = label[np.array(weight_index_med)]
        label_hard = label[np.array(weight_index_hard)]

        sta_easy = []
        sta_med = []
        sta_hard = []
        for i in range(101):
            easy_id = np.where(label_easy == i)[0]
            med_id = np.where(label_med == i)[0]
            hard_id = np.where(label_hard == i)[0]
            sta_easy.append(len(easy_id))
            sta_med.append(len(med_id))
            sta_hard.append(len(hard_id))

        import ipdb;

        ipdb.set_trace()

        np.save('./weights/food101/weight_index_easy.npy', weight_index_easy)
        np.save('./weights/food101/weight_index_med.npy', weight_index_med)
        np.save('./weights/food101/weight_index_hard.npy', weight_index_hard)













































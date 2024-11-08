import os
import torch.nn as nn
import torch

torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch.nn.functional as F
import utils
import skfuzzy as fuzz
import data_loader as loader
from metrics import cal_clustering_metric
from sklearn.metrics import adjusted_rand_score,accuracy_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import silhouette_score,calinski_harabasz_score
import torchvision.transforms as transforms
from PIL import Image
import scipy.io
import scipy.io as scio
from skimage import io, color
from scipy.sparse import coo_matrix
from functions import get_dim, forward, comp_simi
from util import WeightedBCE
from AutomaticWeighted import AutomaticWeightedLoss
from keras.preprocessing.image import ImageDataGenerator

def clustering_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in zip(ind[0],ind[1])])*1.0/Y_pred.size

def purity_score(y_true, y_pred):
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    num_true = len(unique_true)
    num_pred = len(unique_pred)

    contingency_matrix = np.zeros((num_true, num_pred), dtype=np.int64)

    for i in range(len(y_true)):
        true_label = np.where(unique_true == y_true[i])[0][0]
        pred_label = np.where(unique_pred == y_pred[i])[0][0]
        contingency_matrix[true_label, pred_label] += 1

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X,transform=None):
        self.X = X
        self.transform = transform
    def __getitem__(self, idx):
        img = self.X[:, idx]

        #img_tensor = torch.tensor(np.array(img), dtype=torch.float32)

        if self.transform is not None:
            # print(img)
            img = img.cpu().numpy().reshape(32, 32)
            image_array = (img * 255).astype(np.uint8)
            img = Image.fromarray(image_array)
            img_tensor = self.transform(img)
            #Image.fromarray((img_tensor*255).reshape(32,32).numpy().astype(np.uint8)).show()
            img = torch.tensor(np.array(img_tensor), dtype=torch.float32).view(-1)

        #img_tensor = img_tensor.reshape(1024)/255
        #numpy_array = img_tensor.numpy().astype(np.uint8)
        #img = Image.fromarray(numpy_array)


        #img_tensor.show()

        img=img.to('cuda')
        return img, idx
        #return self.X[:, idx], idx

    def __len__(self):
        return self.X.shape[1]

# class PretrainDoubleLayer(torch.nn.Module):
#     def __init__(self, X, dim, device, act, batch_size=128, lr=10**-3):
#         super(PretrainDoubleLayer, self).__init__()
#         self.X = X
#         self.dim = dim
#         self.lr = lr
#         self.device = device
#         self.enc = torch.nn.Linear(X.shape[0], self.dim)
#         self.dec = torch.nn.Linear(self.dim, X.shape[0])
#         self.batch_size = batch_size
#         self.act = act
#
#     def forward(self, x):
#         # print('预训练',self.lr)
#         if self.act is not None:
#             z = self.act(self.enc(x))
#             return z, self.act(self.dec(z))
#         else:
#             z = self.enc(x)
#             return z, self.dec(z)
#
#     def _build_loss(self, x, recons_x):
#         size = x.shape[0]
#         return torch.norm(x-recons_x, p='fro')**2 / size
#
#     def run(self,is_transform=False):
#         self.to(self.device)
#         transform_train = transforms.Compose([
#             transforms.ToTensor(),
#             #transforms.RandomResizedCrop(size=(32, 32)),
#             #transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(degrees=15),
#             transforms.RandomErasing(p=1, scale=(0.02, 0.40), ratio=(0.3, 3.3))
#             #transforms.RandomErasing(p=1, scale=(0.02, 0.03), ratio=(0.3, 0.3), value=(254 / 255, 0, 0))
#             # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         if is_transform==False:
#             transform_train = None
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#
#         train_loader = torch.utils.data.DataLoader(Dataset(self.X,transform=transform_train), batch_size=self.batch_size, shuffle=True)
#         loss = 0
#         for epoch in range(500):
#             for i, batch in enumerate(train_loader):
#                 x, _ = batch
#                 optimizer.zero_grad()
#                 _, recons_x = self(x)
#                 loss = self._build_loss(x, recons_x)
#                 loss.backward()
#                 optimizer.step()
#             print('epoch-{}: loss={}'.format(epoch, loss.item()))
#         Z, _ = self(self.X.t())
#         return Z.t()

class PretrainDoubleLayer(torch.nn.Module):
    def __init__(self, X, dim, device, act, batch_size=128, lr=10**-3):
        super(PretrainDoubleLayer, self).__init__()
        self.X = X
        self.dim = dim
        self.lr = lr
        self.device = device
        self.enc = torch.nn.Linear(X.shape[0], self.dim)
        self.dec = torch.nn.Linear(self.dim, X.shape[0])
        self.batch_size = batch_size
        self.act = act

    def forward(self, x):
        # print('预训练',self.lr)
        if self.act is not None:
            x = self.enc(x)
            z = self.act(x)
            reconx = self.dec(z)
            reconx = self.act(reconx)
            return z, reconx
        else:
            z = self.enc(x)
            return z, self.dec(z)

    def _build_loss(self, x, recons_x):
        size = x.shape[0]
        return torch.norm(x-recons_x, p='fro')**2 / size

    def run(self,is_transform=False):
        self.to(self.device)
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomErasing(p=1, scale=(0.02, 0.40), ratio=(0.3, 3.3))
        ])
        if is_transform==False:
            transform_train = None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        train_loader = torch.utils.data.DataLoader(Dataset(self.X,transform=transform_train), batch_size=self.batch_size, shuffle=True)
        loss = 0
        for epoch in range(500):
            for i, batch in enumerate(train_loader):
                x, _ = batch
                optimizer.zero_grad()
                _, recons_x = self(x)
                loss = self._build_loss(x, recons_x)
                loss.backward()
                optimizer.step()
            print('epoch-{}: loss={}'.format(epoch, loss.item()))
        Z, _ = self(self.X.t())
        return Z.t()


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.BN1 = torch.nn.BatchNorm1d(128, eps=1e-05, track_running_stats=False)
        self.fc2 = torch.nn.Linear(128, 64)
        self.BN2 = torch.nn.BatchNorm1d(64, eps=1e-05, track_running_stats=False)
        self.fc3 = torch.nn.Linear(64, 40)
        self.BN3 = torch.nn.BatchNorm1d(40, eps=1e-05, track_running_stats=False)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        # x = F.relu(self.BN1(x))
        x = self.fc2(x)
        # x = F.relu(x)
        # x = F.relu(self.BN2(x))
        x = self.fc3(x)  # 应用 fc3
        # x = self.BN3(x)  # 然后是 BN3
        x = torch.sigmoid(x)
        # x = torch.nn.functional.softmax(x, dim=1)  # 最后应用 softmax
        return x

class DeepMultiviewFuzzyKMeans(torch.nn.Module):
    def __init__(self, X, labels, layers=None, lam=1.0, lam2=1.0, fuzziness=1.0, lr=10**-3, device="cuda", batch_size=128,num_views=2):
        super(DeepMultiviewFuzzyKMeans, self).__init__()
        if layers is None:
            layers = [X.shape[0], 500, 300]
        # if device is None:
        #     device = torch.device('cuda')
        self.layers = layers
        self.device = device
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        self.X = X.to(device)

        self.labels = labels
        self.lam = lam
        self.lam2= lam2
        self.fuzziness = fuzziness
        self.batch_size = batch_size
        self.n_clusters = len(np.unique(self.labels))
        self.lr = lr
        self.num_views = num_views
        self.alpha_list = np.ones(self.num_views)
        self._build_up()
        self.discriminator = Discriminator(self.layers[2])

    def _build_up(self):
        self.act = torch.tanh
        # self.act = torch.relu
        # self.act = torch.nn.functional.softmax
        self.enc1 = torch.nn.Linear(self.layers[0], self.layers[1])
        self.enc2 = torch.nn.Linear(self.layers[1], self.layers[2])
        self.dec1 = torch.nn.Linear(self.layers[2], self.layers[1])
        self.dec2 = torch.nn.Linear(self.layers[1], self.layers[0])

    def forward(self, x):
        # print('训练', self.lr)
        z = self.act(self.enc1(x))
        z = self.act(self.enc2(z))
        disc_output = self.discriminator(z)
        recons_x = self.act(self.dec1(z))
        recons_x = self.act(self.dec2(recons_x))
        return z, recons_x, disc_output

    # def _build_loss(self, x,multi_view_bdata, u, recons_x):
    #
    #     size = multi_view_bdata[0].shape[0]
    #     #term 1
    #     loss = 1/2 * torch.norm(x - recons_x, p='fro') ** 2 / size
    #     t = u ** self.fuzziness  # t: m * c
    #     #term 2
    #     for i in range(self.num_views):
    #         multi_view_bdata[i] = multi_view_bdata[i].to(torch.float64)
    #         distances = utils.distance(multi_view_bdata[i].t(), self.multi_centroids[i])  # need update
    #         loss += (self.lam / 2 * self.alpha_list[i] * torch.trace(distances.t().matmul(t)) / size)
    #     #term 3
    #     loss += self.lam2 * (self.enc1.weight.norm()**2 + self.enc1.bias.norm()**2) / size
    #     loss += self.lam2 * (self.enc2.weight.norm()**2 + self.enc2.bias.norm()**2) / size
    #     loss += self.lam2 * (self.dec1.weight.norm()**2 + self.dec1.bias.norm()**2) / size
    #     loss += self.lam2 * (self.dec2.weight.norm()**2 + self.dec2.bias.norm()**2) / size
    #     return loss
    def _build_loss(self, z, x,  u, recons_x):
        size = x.shape[0]
        loss = 1/2 * torch.norm(x - recons_x, p='fro') ** 2 / size
        t = self.fuzziness*u  # t: m * c
        # print('444444444444444444',z,self.center)

        z = z.to(torch.float64)
        distances = utils.distance(z.t(), self.center)
        loss += (self.lam / 2 * torch.trace(distances.t().matmul(t)) / size)
        loss += 10**-5 * (self.enc1.weight.norm()**2 + self.enc1.bias.norm()**2) / size
        loss += 10**-5 * (self.enc2.weight.norm()**2 + self.enc2.bias.norm()**2) / size
        loss += 10**-5 * (self.dec1.weight.norm()**2 + self.dec1.bias.norm()**2) / size
        loss += 10**-5 * (self.dec2.weight.norm()**2 + self.dec2.bias.norm()**2) / size
        return loss

    def fcm_centers(self,data, membership, m=2.0):
        # data: nxd
        centroids = torch.matmul(data.T, membership ** m) / (
                    torch.sum(membership ** m, dim=0, keepdim=True) + 0 * torch.finfo(torch.float64).eps)

        return centroids

    def normalize_columns(self,columns):
        # broadcast sum over columns
        normalized_columns = columns / (torch.sum(columns, dim=1, keepdim=True))

        return normalized_columns
    def normalize_power_columns(self,x, exponent):

        x = x.to(torch.float64)

        # values in range [0, 1]
        xmax = torch.max(x, dim=1, keepdim=True)
        x = x / xmax[0]

        # values in range [eps, 1]
        x = torch.clamp(x, min=torch.finfo(x.dtype).eps)
        if exponent < 0:
            # values in range [1, 1/eps]
            xmin = torch.min(x, dim=1, keepdim=True)

            x = x / xmin[0]

            x = x ** exponent
        else:
            # values in range [eps**exponent, 1] where exponent >= 0
            x = x ** exponent

        result = self.normalize_columns(x)

        return result
    def fcm_init_membership(self,X, membership, num_clusters, fuzziness=2.0):
        # Initialize the membership matrix
        #dxn
        num_samples = X.shape[1]
        # 1.Update the cluster centers and calculate distances
        X = X.to(torch.float64)
        self.center = self.fcm_centers(X.T, membership, fuzziness)

        distances = torch.cdist(X.T, self.center.T,2)  # Transpose centroids
        # print('aaaaaaa',torch.sum(self.center))

        # 2.Update the membership matrix
        new_membership = self.normalize_power_columns(distances, -2 / (fuzziness - 1))

        return new_membership

    # def fcm_multiview_0(self,multiview_data, membership, num_clusters, fuzziness=2.0):
    #     # Initialize the membership matrix
    #     num_samples = multiview_data[0].shape[1]
    #     self.multi_centroids = []
    #     multi_distances = []
    #     aggr_distances = 0
    #     # 1.Update the cluster centers and calculate distances
    #     for i in range(len(multiview_data)):
    #         multiview_data[i] = multiview_data[i].to(torch.float64)
    #         center_i = self.fcm_centers(multiview_data[i].T, membership, fuzziness)
    #         self.multi_centroids.append(center_i)
    #
    #         distances = torch.cdist(multiview_data[i].T, center_i.T,2)  # Transpose centroids
    #         distances = distances / np.sqrt(multiview_data[i].shape[0])  # normalized feature distances
    #         #print('multi_view_data:',i,np.sqrt(multiview_data[i].shape[0]) )
    #         aggr_distances = aggr_distances + self.alpha_list[i] * distances ** 2
    #         multi_distances.append(distances)

        # 2.Update the membership matrix
        new_membership = self.normalize_power_columns(aggr_distances, -1 / (fuzziness - 1))

        obj = 0
        # 3.update alpha_list
        for i in range(len(multiview_data)):
            self.alpha_list[i] = 1. / (torch.sqrt(((new_membership ** fuzziness) * (multi_distances[i]) ** 2).sum()))
            obj = obj + torch.sqrt(((new_membership ** fuzziness) * (multi_distances[i]) ** 2).sum())
        # print(((new_membership ** m) * (multi_distances[0])**2).sum().item(),((new_membership ** m) * (multi_distances[1])**2).sum().item(),obj.item())
        membership = new_membership
        print(self.alpha_list)
        return membership, self.multi_centroids, obj


    def run(self):
        self.to(self.device)
        #1. init membership on the original data x

        membership = torch.rand(self.X.shape[1], self.n_clusters, requires_grad=False)
        membership = torch.nn.functional.softmax(membership, dim=1)
        membership = membership.to(torch.float64)
        membership = membership.to("cuda")
        self.center = 0

        # self.X=self.X.to('cuda')
        # self.X = X.cpu().numpy()
        # self.membership = self.fcm_init_membership(self.X,membership,self.n_clusters,fuzziness=self.fuzziness)
        self.pretrain()
        Z, recons_X, disc_output = self(self.X.t())
        Z = Z.t().detach()
        recons_X = recons_X.t().detach()

        crit_graph = nn.BCEWithLogitsLoss().cuda()
        crit_label = WeightedBCE().cuda()
        crit_c = nn.CrossEntropyLoss().cuda()
        # img = self.X[:, 0]
        # img = np.array(img).reshape(32, 32)
        # image_array = (img * 255).astype(np.uint8)
        # img = Image.fromarray(image_array)
        # img.show()
        #
        # img = recons_X[:, 0]
        # img = np.array(img).reshape(32, 32)
        # image_array = (img * 255).astype(np.uint8)
        # img = Image.fromarray(image_array)
        # img.show()
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.18,
            height_shift_range=0.18,
            # channel_shift_range=0.1,
            horizontal_flip=True,
            rescale=0.95,
            zoom_range=[0.85, 1.15])


        self.membership = self.fcm_init_membership(Z,membership,self.n_clusters,fuzziness=self.fuzziness)

        # print('1111111111111111',self.membership.shape,'22222222222222222',disc_output.shape)
        # print(self.membership)

        # self.membership, _, _ = self.fcm_multiview_0([self.X,Z,recons_X], self.membership, self.n_clusters, fuzziness=self.fuzziness)#multiview input
        #self.membership, _, _ = self.fcm_multiview_0([Z], self.membership, self.n_clusters, fuzziness=self.fuzziness)#multiview input

        #idx = random.sample(list(range(Z.shape[1])), self.n_clusters)
        #self.centroids = Z[:, idx] + 10 ** -6
        #self._update_U(Z)
        # D = self._update_D(Z)
        # self.clustering(D, Z)
        print('Starting training......')
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        train_loader = torch.utils.data.DataLoader(Dataset(self.X, transform=transform_train), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        loss = 0
        a=[]
        b=[]
        c=[]

        for epoch in range(100):
            #2.train AE
            # print('1111111111111111',torch.sum(self.center))
            # print('333333333333333333',self.membership)
            # if epoch>2:s
                # print('666666666666666666',self.membership.shape,self.center.shape)
            Z1= (Z+1)/2
            Z1= Z1.t()
            # print('zzzzzzzz',Z1.shape)
            similarity, labels, weights = comp_simi(disc_output)  ##分别是相似矩阵【样本数量，样本数量】，标签矩阵【样本数量，簇】，权重矩阵【样本数量】

            # print('111111111',similarity.shape)
            mask = similarity.ge(0.9)  ##这个就是矩阵W
            self.membership = self.fcm_init_membership(Z, self.membership, self.n_clusters, fuzziness=self.fuzziness)
            # print(mask)
            # print('1111111111111111111111',weights)
            # aa
            for j in range(0,1):
                for i, batch in enumerate(train_loader):
                    x, idx = batch
                    # print('6666666',x.shape)
                    # reshaped_data = x.reshape(x.shape[0],1,32, 32)
                    # reshaped_data = reshaped_data.cpu().numpy()
                    # reshaped_data = np.transpose(reshaped_data, (0, 2, 3, 1))

                    # reshaped_data.to('cpu')
                    # sign = 0
                    # for X_batch_i in datagen.flow(reshaped_data, batch_size=self.batch_size, shuffle=False):
                    #
                    #     aug_input_bs = torch.from_numpy(X_batch_i)
                    #     aug_input_bs = aug_input_bs.float()
                    #     aug_input_batch_var = torch.autograd.Variable(aug_input_bs.cuda())
                    #     X_batch_i = aug_input_batch_var.to('cpu')
                    #     X_batch_i = np.transpose(X_batch_i, (0, 3, 1, 2))
                    #     X_batch_i = X_batch_i.to('cuda')
                    #
                    #     x = X_batch_i.reshape(x.shape[0],-1)
                    optimizer.zero_grad()
                    optimizer_discriminator.zero_grad()
                    # print('输入',x.shape)
                    z, recons_x, disc_output1 = self(x)
                    # print('1111111111111',(z+1)/2)
                    #d = D[idx, :]
                    #print(idx)
                    u = self.membership[idx, :]
                    address = idx.tolist()
                    # print('222', mask,mask.shape)
                    # print('77777', mask[address, :].shape)
                    mask_target = mask[address, :][: ,address].float()
                    # print('222', mask)
                    # print('333', mask_target)
                    # print('333333',labels.shape,type(address),type(address[0]))
                    out_target = labels[address, :]  ##标签矩阵
                    # print('ooooooooo',out_target.shape)
                    weights_batch = weights[address]  ##权重矩阵
                    simi_batch, labels_batch, weigths_tmp = comp_simi(disc_output1)
                    simi_batch = simi_batch / torch.max(simi_batch)
                    # print('66666666666',simi_batch.shape,mask_target.shape)
                    _graph = crit_graph(simi_batch, mask_target)  ##伪图 【相似矩阵，01矩阵W】[32,32][32,32]
                    # print('output0000000000000000000',u,out_target,weights_batch)
                    _label = crit_label(disc_output1, out_target, weights_batch)  ##伪标签，输入【网络的输出，标签矩阵，权重矩阵】([32,10],[32,10],[32]
                    _mview = self._build_loss(z, x, u, recons_x)
                    _label.requires_grad_()
                    _graph.requires_grad_()
                    # print('88888888888888888', _label,_graph)
                     # multiview input
                    _mview3 =  100*_graph + 10*_label
                    # print('88888888888888888', _mview3,_graph,_label)
                    loss = _mview3
                    # loss.requires_grad_()
                    # loss = self._build_loss(x,[x,z,recons_x], u, recons_x) #multiview input
                    #loss = self._build_loss(x,[z], u, recons_x) #multiview input
                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer.step()

                    #
                    # z, recons_x, disc_output1 = self(x)
                    # # _mview = self._build_loss(z, x, u, recons_x)
                    # _mview = self._build_loss(z, x, u, recons_x)
                    # _mview.backward()
                    # optimizer.step()
                    # print('222222222222222',X_batch_i.shape,loss)
                    # sign += 1
                    # if sign > 1:
                    #     break

            print('loss-{}'.format( loss))
            #3. update membership and alpha_list
            Z, recons_X,disc_output = self(self.X.t())
            Z = Z.t().detach()
            recons_X = recons_X.t().detach()
            # print('11111111111111',disc_output)

            # D = self._update_D(Z)
            # for i in range(20):
            # self.membership,_,obj= self.fcm_multiview_0([self.X,Z,recons_X],self.membership,self.n_clusters,fuzziness=self.fuzziness)#multiview input
            # self.membership = self.fcm_init_membership(Z, self.membership, self.n_clusters, fuzziness=self.fuzziness)
            #self.membership,_,obj= self.fcm_multiview_0([Z],self.membership,self.n_clusters,fuzziness=self.fuzziness)#multiview input
            Z1 = (Z + 1) / 2
            Z1 = Z1.t()
            # print('22222222',disc_output)
            shu, y_pred = self.membership.max(dim=1)
            y_pred = y_pred.detach().cpu() + 1
            # print('yyyyyyyyyyyyyyyyyyyyyy',shu)
            y_pred = y_pred.numpy()
            acc = clustering_acc(y_pred,self.labels)
            nmi = normalized_mutual_info_score(y_pred, self.labels)
            purity = purity_score(self.labels, y_pred)
            a.append(acc)
            b.append(nmi)
            c.append(purity)
            #print('epoch-{}, loss={}, ACC={}, NMI={}'.format(epoch, loss.item(), acc, nmi))
            print('epoch-{}, loss={}, obj={},ACC={}, NMI={}, Purity={}'.format(epoch, loss.item(), None, acc, nmi,purity))
            #print('epoch-{}, obj={}, ACC={}, NMI={}, Purity={}'.format(epoch, obj.item(), acc, nmi,purity))
        # return acc,nmi,purity
        return max(a),max(b),max(c)

    def pretrain(self):
        string_template = 'Start pretraining-{}......'
        print(string_template.format(1))
        pre1 = PretrainDoubleLayer(self.X, self.layers[1], self.device, self.act, lr=self.lr*100000)
        Z = pre1.run(is_transform=False)
        self.enc1.weight = pre1.enc.weight
        self.enc1.bias = pre1.enc.bias
        self.dec2.weight = pre1.dec.weight
        self.dec2.bias = pre1.dec.bias
        print(string_template.format(2))
        pre2 = PretrainDoubleLayer(Z.detach(), self.layers[2], self.device, self.act, lr=self.lr*100000)
        pre2.run(is_transform=False)
        self.enc2.weight = pre2.enc.weight
        self.enc2.bias = pre2.enc.bias
        self.dec1.weight = pre2.dec.weight
        self.dec1.bias = pre2.dec.bias
        #
        # pre3 = PretrainDoubleLayer(Z.detach(), self.layers[3], self.device, self.act, lr=self.lr*100000)
        # pre3.run(is_transform=False)
        # self.enc3.weight = pre3.enc.weight
        # self.enc3.bias = pre3.enc.bias
        # self.dec3.weight = pre3.dec.weight
        # self.dec3.bias = pre3.dec.bias
        #


if __name__ == '__main__':
    import data_loader as loader

    #torch.manual_seed(123456)  # 设置随机种子
    # data, labels = loader.load_data(loader.JAFFE)
    # data, labels = loader.load_UMIST()
    # data, labels = loader.load_cifar10()
    # data, labels = loader.load_USPS()
    # data, labels = loader.load_YALE()
    data, labels = loader.load_ORL()

    data = data.T
    acc_list,nmi_list,purity_list = [],[],[]

    for i in range(10):
        lam = 0.001
        lam2 = 50
        fuzziness = 1.14
        lr = 1e-08
        print('lam={} lam2={} fuzziness={}'.format(lam,lam2,fuzziness))
        dfkm = DeepMultiviewFuzzyKMeans(data, labels, [data.shape[0], 256, 128], lam=lam, lam2=lam2, fuzziness = fuzziness, batch_size=128, lr=lr,num_views=3)
        acc,nmi,purity = dfkm.run()
        acc_list.append(acc),nmi_list.append(nmi),purity_list.append(purity)


    aa = np.array(acc_list)
    bb = np.array(nmi_list)
    cc = np.array(purity_list)

    aa1 =  [x * 100 for x in aa]
    bb1 =  [x * 100 for x in bb]
    cc1 =  [x * 100 for x in cc]


    variance = np.var(aa1)
    variance1 = np.var(bb1)
    variance2 = np.var(cc1)

    print(variance, variance1,variance2)
    print(aa1, bb1, cc1)


    print("mean_acc={},mean_nmi={},mean_purity={}\n".format(np.mean(acc_list),np.mean(nmi_list),np.mean(purity_list)))



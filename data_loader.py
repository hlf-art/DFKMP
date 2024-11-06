import scipy.io as scio
import numpy as np
import scipy.io as sio


YALE = 'Yale'
UMIST = 'UMIST'
ORL = 'ORL'
COIL20 = 'COIL20'
YALEB_SDSIFT = 'YaleB_DSIFT'
JAFFE = 'JAFFE'
MNIST_DSIFT = 'mnist_DSIFT'
PALM = 'Palm'
USPS = 'USPSdata_20_uni'
TOY_THREE_RINGS = 'three_rings'
MNIST_TEST = 'mnist_test'
YEAST = 'yeast_uni'
SEGMENT = 'segment_uni'
NEWS = '20news_uni'
WEBK = 'WebKB_wisconsin_uni'
TEXT = 'text1_uni'
GLASS = 'glass_uni'
ISOLET = 'Isolet'
CORA = 'cora'


# def load_cifar10():
#     path='data/cifar-10-batches-mat'
#     data1=scio.loadmat(path+'/data_batch_1.mat')
#     data2=scio.loadmat(path+'/data_batch_2.mat')
#     data3=scio.loadmat(path+'/data_batch_3.mat')
#     data4=scio.loadmat(path+'/data_batch_4.mat')
#     data5=scio.loadmat(path+'/data_batch_5.mat')
#     data6=scio.loadmat(path+'/test_batch.mat')
#     # 拼接数据
#     x= np.concatenate((data1['data'], data2['data'],data3['data'], data4['data'],data5['data'],data6['data']), axis=0)
#
#     label= np.concatenate((data1['labels'], data2['labels'],data3['labels'], data4['labels'],data5['labels'],data6['labels']), axis=0)
#     label=label.ravel()
#     # print(x.shape ,label.shape)
#     data_dict = {'combined_data': x,'label':label}
#     scipy.io.savemat('cifar.mat', data_dict)
#
#     # x=np.transpose(x)
#     x = x.astype(np.float32)
#     x /= np.max(x)
#
#     # print(x)
#     # print(label)
#     print(x.shape,label.shape)
#
#     return x,label
#
#
#
# def load_cora():
#     path = 'data/cora.mat'
#     data = scio.loadmat(path)
#     labels = data['gnd']
#     labels = np.reshape(labels, (labels.shape[0],))
#     X = data['fea']
#     X = X.astype(np.float32)
#     X /= np.max(X)
#     links = data['W']
#     return X, labels, links


def load_data(name):
    path = '/home/hlf/FCM_pycharm_project/data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['X']
    X = np.transpose(X)
    X = X.astype(np.float32)
    X /= np.max(X)

    return X, labels

def load_JAFFE():
    path = '/home/hlf/FCM_pycharm_project/data/JAFFE.mat'
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['X']
    X = np.transpose(X)
    X = X.astype(np.float32)
    X /= np.max(X)

    return X, labels


##Yale_32x32.mat
def load_YALE():
    path = '/home/hlf/FCM_pycharm_project/data/Yale_32x32.mat'
    data = scio.loadmat(path)
    labels = data['gnd']
    print(labels.shape)
    labels = np.reshape(labels, (labels.shape[0],))

    X = data['fea']
    # X = np.transpose(X)
    X = X.astype(np.float32)
    X /= np.max(X)

    print(X.shape,labels.shape)

    return X, labels




def load_UMIST():
    path = '/home/hlf/FCM_pycharm_project/data/UMIST.mat'
    data = scio.loadmat(path)
    labels = data['gnd']

    print('1111',labels)
    print('2222',labels.shape[0])
    labels = np.reshape(labels, (labels.shape[1],))

    X = data['fea']
    X = np.transpose(X)
    X = X.astype(np.float32)
    X /= np.max(X)

    print(X.shape,labels.shape)

    return X, labels



def load_USPS():
    path = '/home/hlf/FCM_pycharm_project/data/USPSdata_20_uni.mat'
    data = scio.loadmat(path)
    labels = data['Y']

    # print('1111',labels)
    # print('2222',labels.shape[0])
    labels = np.reshape(labels, (labels.shape[0],))

    X = data['X']
    # X = np.transpose(X)
    X = X.astype(np.float32)
    X /= np.max(X)

    print(X.shape,labels.shape)

    return X, labels



def load_ORL():
    path = '/home/hlf/FCM_pycharm_project/data/ORL_32.mat'
    data = scio.loadmat(path)
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['fea']
    # X = np.transpose(X)
    X = X.astype(np.float32)
    X /= np.max(X)

    print(X.shape,labels.shape)

    return X, labels


def load_cifar10():
    path='/home/hlf/FCM_pycharm_project/data/cifar1.mat'
    data=scio.loadmat(path)
    # data2=scio.loadmat(path+'/data_batch_2.mat')
    # data3=scio.loadmat(path+'/data_batch_3.mat')
    # data4=scio.loadmat(path+'/data_batch_4.mat')
    # data5=scio.loadmat(path+'/data_batch_5.mat')
    #
    # # 拼接数据
    # x= np.concatenate((data1['data'], data2['data'],data3['data'], data4['data'],data5['data']), axis=0)
    #
    # label= np.concatenate((data1['labels'], data2['labels'],data3['labels'], data4['labels'],data5['labels']), axis=0)
    labels = data['label']
    print(labels.shape)
    labels = np.reshape(labels, (labels.shape[1],))
    X = data['combined_data']
    # X = np.transpose(X)
    X = X.astype(np.float32)
    X /= np.max(X)


    print(X.shape,labels.shape)
    return X,labels

def load_yalebs():
    path = '/home/hlf/FCM_pycharm_project/data/yalebs.mat'
    data = scio.loadmat(path)
    labels = data['label']
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['fea']
    # X = np.transpose(X)
    X = X.astype(np.float32)
    X /= np.max(X)

    print(X.shape,labels.shape)

    return X, labels
# load_cifar10()
# load_data('JAFFE')
# load_YALE()



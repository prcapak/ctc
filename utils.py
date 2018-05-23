import os, numpy as np
from config import FLAGS
import matplotlib.pyplot as plt



def read_mnist_data(is_train=True):
    data_dir=FLAGS.data_dir
    if is_train:
        fd=open(os.path.join(data_dir,"train-images.idx3-ubyte"))
        loaded=np.fromfile(file=fd,dtype=np.uint8)
        X=loaded[16:].reshape([60000,784]).astype(np.float)

        fd=open(os.path.join(data_dir,"train-labels.idx1-ubyte"))
        loaded=np.fromfile(file=fd,dtype=np.uint8)
        Y=loaded[8:].reshape([60000,]).astype(np.float)
    else:
        fd=open(os.path.join(data_dir,"t10k-images.idx3-ubyte"))
        loaded=np.fromfile(file=fd,dtype=np.uint8)
        X=loaded[16:].reshape([10000,784]).astype(np.float)

        fd=open(os.path.join(data_dir,"t10k-labels.idx1-ubyte"))
        loaded=np.fromfile(file=fd,dtype=np.uint8)
        Y=loaded[8:].reshape([10000,]).astype(np.float)

    X=np.asarray(X)/255.
    return X,Y



def one_hot_encoding(dataY,num_classes=10):
    dataY_vec=np.zeros(shape=[dataY.shape[0],num_classes])
    for i,label in enumerate(dataY):
        dataY_vec[i,int(dataY[i])]=1.0
    return dataY_vec



class mnistSampler(object):
    def __init__(self,is_train=True,
                 noise_type='',
                 incorrect_percent=0.1):

        self.dataX,self.dataY=read_mnist_data(is_train)
        self.dataY_ = np.array(self.dataY)
        if is_train:
            if noise_type=="uniform":
                self.uniform_noise(incorrect_percent)
            elif noise_type=="permutation":
                self.permutation_noise(incorrect_percent)

        self.dataY=one_hot_encoding(self.dataY)
        self.dataY_=one_hot_encoding(self.dataY_)
        self.total_size=self.dataX.shape[0]
        self.cursor=0
        self.loop=0

    def next(self, batch_size):
        if self.cursor+batch_size>self.total_size:
            self.shuffle()
            self.cursor=0
        image=self.dataX[self.cursor:self.cursor+batch_size].reshape([batch_size,784])
        y=self.dataY[self.cursor:self.cursor+batch_size]
        y_ = self.dataY_[self.cursor:self.cursor + batch_size]
        self.cursor+=batch_size
        return image,y,y_

    def shuffle(self):
        self.loop+=1
        seed=np.random.randint(1000000)
        np.random.seed(seed)
        np.random.shuffle(self.dataX)
        np.random.seed(seed)
        np.random.shuffle(self.dataY)
        np.random.seed(seed)
        np.random.shuffle(self.dataY_)

    def uniform_noise(self,incorrect_percent):
        self.dataY_=[]
        p = incorrect_percent / (10 - 1)
        ele_list=np.arange(10)
        for original_y in self.dataY:
            p_list = [p] * 10
            p_list[int(original_y)]=1-(10-1)*p
            new_y=np.random.choice(ele_list,1,p=p_list)
            self.dataY_.append(new_y)
        self.dataY_=np.asarray(self.dataY_)

    def permutation_noise(self,incorrect_percent=0.1,seed=1234):
        np.random.seed(seed)
        permutation=np.random.permutation(np.arange(10))
        p_list=[1-incorrect_percent,incorrect_percent]
        self.dataY_=[]
        for original_y in self.dataY:
            ele_list=[original_y,permutation[int(original_y)]]
            new_y=np.random.choice(ele_list,1,p=p_list)
            self.dataY_.append(new_y)
        self.dataY_=np.asarray(self.dataY_)



def plot_lines_chart(noise_list,acc_list,saveDir):
    colorList = ['b', 'r', 'g', 'c', 'm', 'y']
    fig, axes = plt.subplots(1, 1, figsize=(20, 18),squeeze=False)
    ax=axes[0,0]
    nn, = ax.plot(noise_list,acc_list[0], color=colorList[0])
    nlnn, = ax.plot(noise_list,acc_list[1], color=colorList[1])
    nlnn_true, = ax.plot(noise_list,acc_list[2], color=colorList[2])
    ax.legend([nn, nlnn,nlnn_true], ["NN", "NLNN","NLNN-true"])
    ax.set_xlabel("Noise Fraction")
    ax.set_xticks(noise_list)
    ax.set_xticklabels(noise_list)
    ax.set_ylabel("Classification Accuracy")
    if saveDir is not None:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        fig.savefig(os.path.join(saveDir,"accuracy.png"))
    return
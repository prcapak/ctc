from model_nn import basicNNModel
from model_nlnn import basicNLNNModel
from model_nlnn_true import basicNLNNTrueModel
from run import train_nn,train_nlnn,train_nlnn_true,test
import tensorflow as tf
import numpy as np
from utils import plot_lines_chart



def main(unused_argv):
    noise_list=[i for i in np.arange(0,0.65,0.05)]

    nn = basicNNModel()
    # for nt in ["uniform", "permutation"]:
    #     for ip in noise_list:
    #         train_nn(model=nn,noise_type=nt,incorrect_percent=ip)
    nlnn = basicNLNNModel()
    # for nt in ["uniform", "permutation"]:
    #     for ip in noise_list:
    #         train_nlnn(model=nlnn,noise_type=nt,incorrect_percent=ip)
    nlnn_true = basicNLNNTrueModel()
    # for nt in ["uniform", "permutation"]:
    #     for ip in noise_list:
    #         train_nlnn_true(model=nlnn_true,noise_type=nt,incorrect_percent=ip)

    acc_dict={}
    acc_dict["uniform"]=[[],[],[]]
    acc_dict["permutation"]=[[],[],[]]
    for nt in ["uniform", "permutation"]:
        for ip in noise_list:
            acc_dict[nt][0].append(test(nn,
                model_path="logs/"+str(type(nn).__name__)+"/"+nt+"-"+str(ip)+"/model.ckpt-100"))

            acc_dict[nt][1].append(test(nlnn,
                model_path="logs/" + str(type(nlnn).__name__) + "/" + nt + "-" + str(ip) + "/model.ckpt-6"))

            acc_dict[nt][2].append(test(nlnn_true,
                model_path="logs/" + str(type(nlnn_true).__name__) + "/" + nt + "-" + str(ip) + "/model.ckpt-6"))

    plot_lines_chart(noise_list, acc_list=acc_dict["uniform"], saveDir="graphs/uniform/")
    plot_lines_chart(noise_list, acc_list=acc_dict["permutation"], saveDir="graphs/permutation/")





if __name__=="__main__":
    tf.app.run()
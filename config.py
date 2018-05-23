import tensorflow as tf


# path
tf.app.flags.DEFINE_string("data_dir","MNIST/","Directory for reading data.")
tf.app.flags.DEFINE_string("log_dir","logs/","Directory for saving logs and models.")
tf.app.flags.DEFINE_string("graph_dir","graphs/","Directory for saving graphs.")

# model
tf.app.flags.DEFINE_integer("batch_size",128,"Size for a batch.")
tf.app.flags.DEFINE_float("learning_rate",0.01,"Initial learning rate.")
tf.app.flags.DEFINE_float("momentum",0.5,"Momentum value.")
tf.app.flags.DEFINE_string("hidden_size_list","500-300","List for hidden layers' sizes.")

# train
tf.app.flags.DEFINE_string("load_path_train",None,
                           "Path for loading model when training.")
tf.app.flags.DEFINE_integer("training_epoches",100,"Training epoches for basic NN model.")
tf.app.flags.DEFINE_string("noise_type","uniform","Type of adding noise.")
tf.app.flags.DEFINE_float("incorrect_percent",0.5,"Percentage for adding noise.")
tf.app.flags.DEFINE_integer("save_epoch",10,"Saving model every save epoch.")

tf.app.flags.DEFINE_integer("initial_epoches",100,
                            "Training epoches for neural network when initialization.")
tf.app.flags.DEFINE_integer("m_internal_epoches",50,
                            "Training epoches for neural network at M-step.")
tf.app.flags.DEFINE_integer("em_epoches",8,"Training epoches for total EM process.")

# test
tf.app.flags.DEFINE_string("load_path_test",None,
                           "Path for loading model when testing.")
# tf.app.flags.DEFINE_string("load_path_test","logs/basicNLNNModel/uniform-0.5/model.ckpt-8",
#                            "Path for loading model when testing.")
# tf.app.flags.DEFINE_string("load_path_test","logs/basicNLNNTrueModel/uniform-0.5/model.ckpt-8",
#                            "Path for loading model when testing.")

FLAGS=tf.flags.FLAGS
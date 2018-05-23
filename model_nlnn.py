from config import FLAGS
import tensorflow as tf



class basicNLNNModel(object):

    def __init__(self, batch_size=FLAGS.batch_size,
                 hidden_size_list=[int(s) for s in FLAGS.hidden_size_list.split("-")],
                 learning_rate=FLAGS.learning_rate,
                 momentum=FLAGS.momentum,
                 initial_epoches=FLAGS.initial_epoches,
                 m_internal_epoches=FLAGS.m_internal_epoches):
        self.batch_size = batch_size
        self.hidden_size_list = hidden_size_list
        self.initial_learning_rate = learning_rate
        self.momentum = momentum
        self.initial_epoches = initial_epoches
        self.m_internal_epoches = m_internal_epoches
        self.build_model()

    def build_model(self):
        self.graph=tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope("input_layer"):
                self.X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 784],
                                        name="input_features_placeholder")
                self.y_noise_vec = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 10],
                                                  name="noisy_labels_placeholder")
                self.y_vec = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 10],
                                            name="true_labels_placeholder")
                self.learning_rate = tf.placeholder_with_default(self.initial_learning_rate, shape=[],
                                                                 name="learning_rate_placeholder")
            with tf.variable_scope("hidden_layers"):
                h = self.X
                for s in self.hidden_size_list:
                    h = tf.layers.dense(inputs=h, units=s, activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope("output_layer"):
                self.logits = tf.layers.dense(inputs=h, units=10,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("Accuracy"):
                self.predicted = tf.nn.softmax(self.logits)
                correct_or_not = tf.equal(tf.argmax(self.predicted, axis=-1),
                                          tf.argmax(self.y_noise_vec, axis=-1))
                correct_or_not_true = tf.equal(tf.argmax(self.predicted, axis=-1),
                                               tf.argmax(self.y_vec, axis=-1))
                self.acc_op = tf.reduce_mean(tf.cast(correct_or_not, tf.float32))
                self.acc_op_true = tf.reduce_mean(tf.cast(correct_or_not_true, tf.float32))

            with tf.name_scope("Loss"):
                self.loss_op = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_noise_vec))

            with tf.name_scope("Optimize"):
                self.train_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                           momentum=self.momentum).minimize(self.loss_op)

            with tf.variable_scope("noisy_channel"):
                self.theta_i_j = tf.get_variable(shape=[10, 10],
                                                 initializer=tf.zeros_initializer(),
                                                 dtype=tf.float32, trainable=False,
                                                 name="parameter")

                i_j = tf.matmul(tf.transpose(self.y_vec), self.y_noise_vec)
                i_ = tf.reshape(tf.reduce_sum(self.y_vec, reduction_indices=0), shape=[-1, 1])
                self.theta_i_j_value = i_j / i_

                exp_logits = tf.exp(self.logits)
                b_i = tf.matmul(self.y_noise_vec, tf.transpose(self.theta_i_j))
                b_i = b_i * exp_logits
                b_ = tf.reshape(tf.reduce_sum(b_i, reduction_indices=-1), shape=[-1, 1])
                self.c_t_i_value = b_i / b_
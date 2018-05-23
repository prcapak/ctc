from utils import mnistSampler
import os, numpy as np
from config import FLAGS
import tensorflow as tf



def train_nn(model, load_path=FLAGS.load_path_train,
          training_epoches=FLAGS.training_epoches,
          noise_type=FLAGS.noise_type,
          incorrect_percent=FLAGS.incorrect_percent
          ):
        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    train_log_dir = FLAGS.log_dir + str(type(model).__name__) \
                    + "/" + noise_type + "-" + str(incorrect_percent)

    train_gen = mnistSampler(is_train=True,
                             noise_type=noise_type, incorrect_percent=incorrect_percent)
    num_batches = train_gen.total_size // model.batch_size
    avg_losses = []
    initial_lr = model.initial_learning_rate

    with tf.Session(config=config,graph=model.graph) as sess:

        tf.summary.FileWriter(train_log_dir,graph=sess.graph)
        saver = tf.train.Saver(max_to_keep=5)

        sess.run(tf.global_variables_initializer())
        start = 0
        if isinstance(load_path, str):
            print("Reading checkpoints...")
            saver.restore(sess, load_path)
            start = int(load_path.split("-")[-1])
            print("Loading successfully, global epoch is %s" % start)

        for epoch in range(start, training_epoches):
            if len(avg_losses) > 1:
                if avg_losses[-1] - avg_losses[-2] < -np.log(1.04):
                    initial_lr *= 1.05
                    print("Using higher learning rate...")
                if avg_losses[-1] > avg_losses[-2]:
                    initial_lr *= 0.7
                    print("Using lower learning rate...")
            avg_loss, avg_acc, avg_acc_true = 0., 0., 0.
            for n in range(num_batches):
                batch_x, batch_y, batch_y_ = train_gen.next(model.batch_size)
                _, loss, acc, acc_t = sess.run([model.train_op, model.loss_op,
                                                model.acc_op, model.acc_op_true],
                                               feed_dict={model.X: batch_x, model.y_vec: batch_y,
                                                          model.y_noise_vec: batch_y_,
                                                          model.learning_rate: initial_lr})
                avg_loss += loss
                avg_acc += acc
                avg_acc_true += acc_t

            avg_loss /= num_batches
            avg_acc /= num_batches
            avg_acc_true /= num_batches

            avg_losses.append(avg_loss)
            print("Epoch: [%04d/%04d], Training Loss: %.4f, "
                  "Training Accuracy: %.4f, Training Accuracy True: %.4f"
                  % (epoch + 1, training_epoches, avg_loss, avg_acc, avg_acc_true))
            if (epoch - start + 1) % FLAGS.save_epoch == 0:
                saver.save(sess, save_path=os.path.join(train_log_dir, "model.ckpt")
                           , global_step=epoch + 1)



def train_nlnn(model, load_path=FLAGS.load_path_train,
          em_epoches=FLAGS.em_epoches,
          noise_type=FLAGS.noise_type,
          incorrect_percent=FLAGS.incorrect_percent
          ):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    sess = tf.InteractiveSession(config=config,graph=model.graph)
    sess_1 = tf.InteractiveSession(config=config,graph=model.graph)

    saver=tf.train.Saver(max_to_keep=5)
    train_log_dir = FLAGS.log_dir + str(type(model).__name__) \
                    + "/" + noise_type + "-" + str(incorrect_percent)

    train_gen = mnistSampler(is_train=True,
                             noise_type=noise_type, incorrect_percent=incorrect_percent)
    num_batches = train_gen.total_size // model.batch_size

    tf.summary.FileWriter(train_log_dir,sess.graph)

    start = 0
    if isinstance(load_path, str):
        print("Reading checkpoints...")
        saver.restore(sess, load_path)
        start = int(load_path.split("-")[-1])
        print("Loading successfully, global epoch is %s" % start)

    if load_path is None:
        print("Initializing...")
        initial_lr = model.initial_learning_rate
        avg_losses = []
        sess.run(tf.global_variables_initializer())
        for epoch in range(model.initial_epoches):
            if len(avg_losses) > 1:
                if avg_losses[-1] - avg_losses[-2] < -np.log(1.04):
                    initial_lr *= 1.05
                    print("Using higher learning rate...")
                if avg_losses[-1] > avg_losses[-2]:
                    initial_lr *= 0.7
                    print("Using lower learning rate...")
            avg_loss, avg_acc, avg_acc_true = 0., 0., 0.
            for n in range(num_batches):
                batch_x, batch_y, batch_y_ = train_gen.next(model.batch_size)
                _, loss, acc, acc_true = sess.run([model.train_op, model.loss_op,
                                                   model.acc_op, model.acc_op_true],
                                                  feed_dict={model.X: batch_x,
                                                             model.y_noise_vec: batch_y_,
                                                             model.y_vec: batch_y,
                                                             model.learning_rate: initial_lr})
                avg_loss += loss
                avg_acc += acc
                avg_acc_true += acc_true
            avg_loss /= num_batches
            avg_acc /= num_batches
            avg_acc_true /= num_batches
            avg_losses.append(avg_loss)
            print("Epoch: [%04d/%04d], Training Loss: %.4f, "
                  "Training Accuracy: %.4f, Training Accuracy True: %.4f"
                  % (epoch + 1, model.initial_epoches, avg_loss, avg_acc, avg_acc_true))
        theta_i_j = np.zeros(shape=[10, 10], dtype=np.float32)
        for n in range(num_batches):
            batch_x, batch_y, batch_y_ = train_gen.next(model.batch_size)
            predicted = sess.run(model.predicted, feed_dict={model.X: batch_x})
            theta_i_j_ = sess.run(model.theta_i_j_value, feed_dict={model.y_noise_vec: batch_y_,
                                                                   model.y_vec: predicted})
            theta_i_j += theta_i_j_
        theta_i_j /= num_batches
        model.theta_i_j.load(theta_i_j, sess)
        # print(sess.run(model.theta_i_j)[0])
        saver.save(sess, save_path=os.path.join(train_log_dir, "model.ckpt"), global_step=0)
    print("E-Step,M-Step...")
    for em_epoch in range(start, em_epoches):
        saver.restore(sess_1, save_path=os.path.join(train_log_dir, "model.ckpt-" + str(em_epoch)))
        sess.run(tf.global_variables_initializer())
        print("Updating theta...")
        # update theta
        theta_i_j = np.zeros(shape=[10, 10], dtype=np.float32)
        for n in range(num_batches):
            batch_x, batch_y, batch_y_ = train_gen.next(model.batch_size)
            # if n==0:
            #     print(sess_1.run(model.theta_i_j)[0])
            c_t_i = sess_1.run(model.c_t_i_value, feed_dict={model.X: batch_x,
                                                            model.y_noise_vec: batch_y_})
            theta_i_j_ = sess.run(model.theta_i_j_value, feed_dict={model.y_noise_vec: batch_y_,
                                                                   model.y_vec: c_t_i})
            theta_i_j += theta_i_j_
        theta_i_j /= num_batches
        model.theta_i_j.load(theta_i_j, sess)
        # print(sess.run(model.theta_i_j)[0])
        print("Updating w...")
        # update w
        avg_losses = []
        initial_lr = model.initial_learning_rate
        for epoch in range(model.m_internal_epoches):
            if len(avg_losses) > 1:
                if avg_losses[-1] - avg_losses[-2] < -np.log(1.04):
                    initial_lr *= 1.05
                    print("Using higher learning rate...")
                if avg_losses[-1] > avg_losses[-2]:
                    initial_lr *= 0.7
                    print("Using lower learning rate...")
            avg_loss, avg_acc, avg_acc_true = 0., 0., 0.
            for n in range(num_batches):
                batch_x, batch_y, batch_y_ = train_gen.next(model.batch_size)
                c_t_i = sess_1.run(model.c_t_i_value, feed_dict={model.X: batch_x,
                                                                model.y_noise_vec: batch_y_})
                # print(sess.run(model.theta_i_j)[0])
                _, loss, acc, acc_true = sess.run([model.train_op, model.loss_op,
                                                   model.acc_op, model.acc_op_true],
                                                  feed_dict={model.X: batch_x,
                                                             model.y_noise_vec: c_t_i,
                                                             model.y_vec: batch_y,
                                                             model.learning_rate: initial_lr})
                avg_loss += loss
                avg_acc += acc
                avg_acc_true += acc_true
            avg_loss /= num_batches
            avg_acc /= num_batches
            avg_acc_true /= num_batches
            avg_losses.append(avg_loss)
            print("Epoch: [%04d/%04d], Training Loss: %.4f, "
                  "Training Accuracy: %.4f, Training Accuracy True: %.4f"
                  % (epoch + 1, model.m_internal_epoches, avg_loss, avg_acc, avg_acc_true))

        avg_acc, avg_acc_true = 0., 0.
        for n in range(num_batches):
            batch_x, batch_y, batch_y_ = train_gen.next(model.batch_size)
            acc, acc_true = sess.run([model.acc_op, model.acc_op_true],
                                     feed_dict={model.X: batch_x, model.y_vec: batch_y,
                                                model.y_noise_vec: batch_y_})
            avg_acc += acc
            avg_acc_true += acc_true
        avg_acc /= num_batches
        avg_acc_true /= num_batches
        print("Epoch: [%04d/%04d], Training Accuracy: %.4f, Training Accuracy True: %.4f"
              % (em_epoch + 1, em_epoches, avg_acc, avg_acc_true))

        saver.save(sess, save_path=os.path.join(train_log_dir, "model.ckpt")
                   , global_step=em_epoch + 1)
    sess_1.close()
    sess.close()



def train_nlnn_true(model, load_path=FLAGS.load_path_train,
          em_epoches=FLAGS.em_epoches,
          noise_type=FLAGS.noise_type,
          incorrect_percent=FLAGS.incorrect_percent
          ):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    sess = tf.InteractiveSession(config=config,graph=model.graph)
    sess_1 = tf.InteractiveSession(config=config,graph=model.graph)

    saver=tf.train.Saver(max_to_keep=5)
    train_log_dir = FLAGS.log_dir + str(type(model).__name__) \
                    + "/" + noise_type + "-" + str(incorrect_percent)

    train_gen = mnistSampler(is_train=True,
                             noise_type=noise_type, incorrect_percent=incorrect_percent)
    num_batches = train_gen.total_size // model.batch_size

    train_writer = tf.summary.FileWriter(train_log_dir)
    train_writer.add_graph(sess.graph)

    start = 0
    if isinstance(load_path, str):
        print("Reading checkpoints...")
        saver.restore(sess, load_path)
        start = int(load_path.split("-")[-1])
        print("Loading successfully, global epoch is %s" % start)

    if load_path is None:
        print("Initializing...")
        initial_lr = model.initial_learning_rate
        avg_losses = []
        sess.run(tf.global_variables_initializer())
        for epoch in range(model.initial_epoches):
            if len(avg_losses) > 1:
                if avg_losses[-1] - avg_losses[-2] < -np.log(1.04):
                    initial_lr *= 1.05
                    print("Using higher learning rate...")
                if avg_losses[-1] > avg_losses[-2]:
                    initial_lr *= 0.7
                    print("Using lower learning rate...")
            avg_loss, avg_acc, avg_acc_true = 0., 0., 0.
            for n in range(num_batches):
                batch_x, batch_y, batch_y_ = train_gen.next(model.batch_size)
                _, loss, acc, acc_true = sess.run([model.train_op, model.loss_op,
                                                   model.acc_op, model.acc_op_true],
                                                  feed_dict={model.X: batch_x,
                                                             model.y_noise_vec: batch_y_,
                                                             model.y_vec: batch_y,
                                                             model.learning_rate: initial_lr})
                avg_loss += loss
                avg_acc += acc
                avg_acc_true += acc_true
            avg_loss /= num_batches
            avg_acc /= num_batches
            avg_acc_true /= num_batches
            avg_losses.append(avg_loss)
            print("Epoch: [%04d/%04d], Training Loss: %.4f, "
                  "Training Accuracy: %.4f, Training Accuracy True: %.4f"
                  % (epoch + 1, model.initial_epoches, avg_loss, avg_acc, avg_acc_true))
        theta_i_j = np.zeros(shape=[10, 10], dtype=np.float32)
        for n in range(num_batches):
            batch_x, batch_y, batch_y_ = train_gen.next(model.batch_size)
            theta_i_j_ = sess.run(model.theta_i_j_value, feed_dict={model.y_noise_vec: batch_y_,
                                                                   model.y_vec: batch_y})
            theta_i_j += theta_i_j_
        theta_i_j /= num_batches
        np.save(train_log_dir+"/theta.npy",theta_i_j)
        model.theta_i_j.load(theta_i_j, sess)
        # print(sess.run(model.theta_i_j)[0])
        saver.save(sess, save_path=os.path.join(train_log_dir, "model.ckpt"), global_step=0)
    print("E-Step,M-Step...")
    for em_epoch in range(start, em_epoches):
        saver.restore(sess_1, save_path=os.path.join(train_log_dir, "model.ckpt-" + str(em_epoch)))
        sess.run(tf.global_variables_initializer())
        print("Loading theta...")
        # load theta
        theta_i_j = np.load(train_log_dir+"/theta.npy")
        model.theta_i_j.load(theta_i_j, sess)
        # print(sess.run(model.theta_i_j)[0])
        print("Updating w...")
        # update w
        avg_losses = []
        initial_lr = model.initial_learning_rate
        for epoch in range(model.m_internal_epoches):
            if len(avg_losses) > 1:
                if avg_losses[-1] - avg_losses[-2] < -np.log(1.04):
                    initial_lr *= 1.05
                    print("Using higher learning rate...")
                if avg_losses[-1] > avg_losses[-2]:
                    initial_lr *= 0.7
                    print("Using lower learning rate...")
            avg_loss, avg_acc, avg_acc_true = 0., 0., 0.
            for n in range(num_batches):
                batch_x, batch_y, batch_y_ = train_gen.next(model.batch_size)
                c_t_i = sess_1.run(model.c_t_i_value, feed_dict={model.X: batch_x,
                                                                model.y_noise_vec: batch_y_})
                # print(sess.run(model.theta_i_j)[0])
                _, loss, acc, acc_true = sess.run([model.train_op, model.loss_op,
                                                   model.acc_op, model.acc_op_true],
                                                  feed_dict={model.X: batch_x,
                                                             model.y_noise_vec: c_t_i,
                                                             model.y_vec: batch_y,
                                                             model.learning_rate: initial_lr})
                avg_loss += loss
                avg_acc += acc
                avg_acc_true += acc_true
            avg_loss /= num_batches
            avg_acc /= num_batches
            avg_acc_true /= num_batches
            avg_losses.append(avg_loss)
            print("Epoch: [%04d/%04d], Training Loss: %.4f, "
                  "Training Accuracy: %.4f, Training Accuracy True: %.4f"
                  % (epoch + 1, model.m_internal_epoches, avg_loss, avg_acc, avg_acc_true))

        avg_acc, avg_acc_true = 0., 0.
        for n in range(num_batches):
            batch_x, batch_y, batch_y_ = train_gen.next(model.batch_size)
            acc, acc_true = sess.run([model.acc_op, model.acc_op_true],
                                     feed_dict={model.X: batch_x, model.y_vec: batch_y,
                                                model.y_noise_vec: batch_y_})
            avg_acc += acc
            avg_acc_true += acc_true
        avg_acc /= num_batches
        avg_acc_true /= num_batches
        print("Epoch: [%04d/%04d], Training Accuracy: %.4f, Training Accuracy True: %.4f"
              % (em_epoch + 1, em_epoches, avg_acc, avg_acc_true))

        saver.save(sess, save_path=os.path.join(train_log_dir, "model.ckpt")
                   , global_step=em_epoch + 1)
    sess_1.close()
    sess.close()



def test(model, model_path=FLAGS.load_path_test):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    test_gen = mnistSampler(is_train=False)
    num_batches = test_gen.total_size // model.batch_size

    with tf.Session(config=config,graph=model.graph) as sess:
        saver = tf.train.Saver(max_to_keep=5)
        print("Loading model...")
        saver.restore(sess, model_path)
        global_step = model_path.split("-")[-1]
        print("Loading successfully, global step is %s" % global_step)

        avg_acc = 0.
        for n in range(num_batches):
            batch_x, batch_y, _ = test_gen.next(model.batch_size)
            acc = sess.run(model.acc_op_true, feed_dict={model.X: batch_x, model.y_vec: batch_y})
            avg_acc += acc
        avg_acc /= num_batches
        print("Testing Accuracy True: %.4f"
              % (avg_acc))
    return avg_acc
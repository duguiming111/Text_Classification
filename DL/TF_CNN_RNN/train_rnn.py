# Author:duguiming
# Description:训练RNN模型
# Date:2019-07-08
import time
import os
import tensorflow as tf
from data_helper.data_process import DataProcess
from config.config import RNNConfig, PathConfig
from model.text_model import TextRNN


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """验证集测试"""
    data_len = len(x_)
    batch_eval = dp.batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


def train():
    start_time = time.time()
    print("Loading training ...")
    X_train, y_train = dp.process_file(pathcnfig.train_dir, word2id, cat2id, rnnconfig.seq_length)
    X_val, y_val = dp.process_file(pathcnfig.val_dir, word2id, cat2id, rnnconfig.seq_length)
    print('Time coat:{:.3f}'.format(time.time() - start_time))

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(pathcnfig.rnn_tensorboard_dir)
    saver = tf.train.Saver()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print("Training and evaling...")
    best_val_accuracy = 0
    last_improved = 0
    require_improvement = 1000
    flag = False

    for epoch in range(rnnconfig.num_epochs):
        batch_train = dp.batch_iter(X_train, y_train, rnnconfig.batch_size)
        start = time.time()
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, rnnconfig.dropout_keep_prob)
            _, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
                                                                                       merged_summary, model.loss,
                                                                                       model.acc], feed_dict=feed_dict)
            if global_step % rnnconfig.print_per_batch == 0:
                end = time.time()
                val_loss, val_accuracy = evaluate(session, X_val, y_val)
                writer.add_summary(train_summaries, global_step)

                # If improved, save the model
                if val_accuracy > best_val_accuracy:
                    saver.save(session, pathcnfig.rnn_best_model_dir)
                    best_val_accuracy = val_accuracy
                    last_improved = global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, "
                      "val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
                    global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                    (end - start) / rnnconfig.print_per_batch, improved_str))
                start = time.time()

            if global_step - last_improved > require_improvement:
                print("No optimization over 1000 steps, stop training")
                flag = True
                break
        if flag:
            break


if __name__ == "__main__":
    pathcnfig = PathConfig()
    rnnconfig = RNNConfig()
    dp = DataProcess()
    filenames = [pathcnfig.train_dir, pathcnfig.val_dir, pathcnfig.test_dir]
    if not os.path.exists(pathcnfig.vocab_dir):
        dp.build_vocab(filenames, pathcnfig.vocab_dir, rnnconfig.vocab_size)

    # 读取词表和类别
    categories, cat2id = dp.read_category()
    words, word2id = dp.read_vocab(pathcnfig.vocab_dir)

    # 转化为向量
    if not os.path.exists(pathcnfig.vector_word_npz):
        dp.export_word2vec_vectors(word2id, pathcnfig.word2vec_save_path, pathcnfig.vector_word_npz)
    rnnconfig.pre_training = dp.get_training_word2vec_vectors(pathcnfig.vector_word_npz)

    model = TextRNN(rnnconfig)
    train()
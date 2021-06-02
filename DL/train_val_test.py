# Author: duguiming
# Description: 训练验证和测试
# Date:2020-3-31
import time
import numpy as np
import tensorflow as tf
from sklearn import metrics

from DL.models.base_config import BaseConfig
from DL.data_helper.data_process import DataProcess

base_config = BaseConfig()
dp = DataProcess(base_config)


def feed_data(model, x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, model, config, x_, y_):
    """验证集测试"""
    data_len = len(x_)
    batch_eval = dp.batch_iter(x_, y_, config.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        if model.name == "Transformer" and batch_len < config.batch_size:
            continue
        feed_dict = feed_data(model, x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


def train(model, config, word2id, cat2id):
    start_time = time.time()
    print("Loading training ...")
    X_train, y_train = dp.process_file(config.train_dir, word2id, cat2id, config.seq_length)
    X_val, y_val = dp.process_file(config.val_dir, word2id, cat2id, config.seq_length)
    print('Time coast:{:.3f}'.format(time.time()-start_time))

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(config.tensorboard_dir)
    saver = tf.train.Saver()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print("Training and evaling...")
    best_val_accuracy = 0
    last_improved = 0
    require_improvement = 1000
    flag = False

    for epoch in range(config.num_epochs):
        batch_train = dp.batch_iter(X_train, y_train, config.batch_size)
        start = time.time()
        for x_batch, y_batch in batch_train:
            if model.name == "Transformer" and len(x_batch) < config.batch_size:
                continue
            feed_dict = feed_data(model, x_batch, y_batch, config.dropout_keep_prob)
            _, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
                                                                                       merged_summary, model.loss,
                                                                                       model.acc], feed_dict=feed_dict)
            if global_step % config.print_per_batch == 0:
                end = time.time()
                val_loss, val_accuracy = evaluate(session, model, config, X_val, y_val)
                writer.add_summary(train_summaries, global_step)

                # If improved, save the model
                if val_accuracy > best_val_accuracy:
                    saver.save(session, config.best_model_dir)
                    best_val_accuracy = val_accuracy
                    last_improved = global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, "
                       "val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
                        global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                        (end - start) / config.print_per_batch, improved_str))
                start = time.time()

            if global_step - last_improved > require_improvement:
                print("No optimization over 1000 steps, stop training")
                flag = True
                break
        if flag:
            break


def test(model, config, word2id, cat2id, categories):
    print("Loading test data...")
    start = time.time()
    x_test, y_test = dp.process_file(config.test_dir, word2id, cat2id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=config.best_model_dir)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, model, config, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = config.batch_size
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    if model.name == "Transformer":
        y_test_cls = np.argmax(y_test[0: config.batch_size*(num_batch-1)], 1)
        y_pred_cls = np.zeros(shape=len(x_test[0: config.batch_size*(num_batch-1)]), dtype=np.int32)  # 保存预测结果
    else:
        y_test_cls = np.argmax(y_test, 1)
        y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        if model.name == "Transformer" and (i + 1) * batch_size > data_len:
             continue
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    end = time.time()
    print("Time usage:", end - start)

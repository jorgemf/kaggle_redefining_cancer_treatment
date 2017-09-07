import tensorflow as tf


def single_label(predictions_batch, one_hot_labels_batch, moving_average=True):
    with tf.variable_scope('metrics'):
        shape = predictions_batch.get_shape().as_list()
        batch_size, num_outputs = shape[0], shape[1]
        # get the most probable label
        predicted_batch = tf.argmax(predictions_batch, axis=1)
        real_label_batch = tf.argmax(one_hot_labels_batch, axis=1)

        # tp, tn, fp, fn
        predicted_bool = tf.cast(tf.one_hot(predicted_batch, depth=num_outputs), dtype=tf.bool)
        real_bool = tf.cast(tf.one_hot(real_label_batch, depth=num_outputs), dtype=tf.bool)
        d = _metrics(predicted_bool, real_bool, moving_average)

        # confusion matrix
        confusion_batch = tf.confusion_matrix(labels=real_label_batch, predictions=predicted_batch,
                                              num_classes=num_outputs)

        if moving_average:
            # calculate moving averages
            confusion_batch = tf.cast(confusion_batch, dtype=tf.float32)
            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            update_op = ema.apply([confusion_batch])
            confusion_matrix = ema.average(confusion_batch)
            d['update_op'] = [d['update_op'], update_op]
        else:
            # accumulative
            confusion_matrix = tf.Variable(tf.zeros([num_outputs, num_outputs], dtype=tf.int32),
                                           name='confusion_matrix', trainable=False)
            confusion_matrix = tf.assign_add(confusion_matrix, confusion_batch)

    d['confusion_matrix'] = confusion_matrix
    return d


def multi_label(prediction_batch, labels_batch, threshold=0.5, moving_average=True):
    with tf.variable_scope('metrics'):
        threshold_graph = tf.constant(threshold, name='threshold')
        zero_point_five = tf.constant(0.5)
        predicted_bool = tf.greater_equal(prediction_batch, threshold_graph)
        real_bool = tf.greater_equal(labels_batch, zero_point_five)
        return _metrics(predicted_bool, real_bool, moving_average)


def _metrics(predicted_bool, real_bool, moving_average):
    predicted_bool_neg = tf.logical_not(predicted_bool)
    real_bool_neg = tf.logical_not(real_bool)
    differences_bool = tf.logical_xor(predicted_bool, real_bool)
    tp = tf.logical_and(predicted_bool, real_bool)
    tn = tf.logical_and(predicted_bool_neg, real_bool_neg)
    fn = tf.logical_and(differences_bool, real_bool)
    fp = tf.logical_and(differences_bool, predicted_bool)
    if moving_average:
        # calculate moving averages
        tp = tf.reduce_sum(tf.cast(tp, tf.float32))
        tn = tf.reduce_sum(tf.cast(tn, tf.float32))
        fn = tf.reduce_sum(tf.cast(fn, tf.float32))
        fp = tf.reduce_sum(tf.cast(fp, tf.float32))
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        update_op = ema.apply([tp, tn, fp, fn])
        tp = ema.average(tp)
        tn = ema.average(tn)
        fp = ema.average(fp)
        fn = ema.average(fn)
    else:
        # accumulative
        tp = tf.reduce_sum(tf.cast(tp, tf.int32))
        tn = tf.reduce_sum(tf.cast(tn, tf.int32))
        fn = tf.reduce_sum(tf.cast(fn, tf.int32))
        fp = tf.reduce_sum(tf.cast(fp, tf.int32))
        tp_v = tf.Variable(0, dtype=tf.int32, name='true_positive', trainable=False)
        tn_v = tf.Variable(0, dtype=tf.int32, name='true_negative', trainable=False)
        fp_v = tf.Variable(0, dtype=tf.int32, name='false_positive', trainable=False)
        fn_v = tf.Variable(0, dtype=tf.int32, name='false_negative', trainable=False)
        tp = tf.cast(tf.assign_add(tp_v, tp), dtype=tf.float32)
        tn = tf.cast(tf.assign_add(tn_v, tn), dtype=tf.float32)
        fp = tf.cast(tf.assign_add(fp_v, fp), dtype=tf.float32)
        fn = tf.cast(tf.assign_add(fn_v, fn), dtype=tf.float32)
        update_op = []

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fall_out = fp / (tn + fp)
    f1_score = tp * 2 / (tp * 2 + fp + fn)
    # remove NaNs and set them to 0
    zero = tf.constant(0, dtype=tf.float32)
    precision = tf.cond(tf.equal(tp, 0.0), lambda: zero, lambda: precision)
    recall = tf.cond(tf.equal(tp, 0.0), lambda: zero, lambda: recall)
    accuracy = tf.cond(tf.equal(tp + tn, 0.0), lambda: zero, lambda: accuracy)
    fall_out = tf.cond(tf.equal(fp, 0.0), lambda: zero, lambda: fall_out)
    f1_score = tf.cond(tf.equal(tp, 0.0), lambda: zero, lambda: f1_score)

    # add to tensorboard
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('precision', precision)
    tf.summary.scalar('recall', recall)
    tf.summary.scalar('fall-out', fall_out)
    tf.summary.scalar('f1-score', f1_score)
    tf.summary.scalar('true positive', tp)
    tf.summary.scalar('true negative', tn)
    tf.summary.scalar('false positive', fp)
    tf.summary.scalar('false negative', fn)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fall-out': fall_out,
        'f1-score': f1_score,
        'true positive': tp,
        'true negative': tn,
        'false positive': fp,
        'false negative': fn,
        'update_op': update_op
    }


with graph:
    # 1st arg of scalar_summary is just a 'Name tag' (to appear on the TensorBoard)
    tf.scalar_summary(loss.op.name, loss)
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(
        FLAGS.train_dir,
        graph_def=sess.graph_def)

    saver = tf.train.Saver()

with session:
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, step)
    # save a (bakcup) model
    
    saver.save(sess, FLAGS.train_dir, global_step=step)
    # saver.restore(sess, FLAGS.train_dir)

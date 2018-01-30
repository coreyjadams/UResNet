# Larcv imports:
import ROOT
from larcv import larcv
larcv.ThreadProcessor
from larcv.dataloader2 import larcv_threadio

# Misc python imports:
import numpy
import matplotlib
import matplotlib.pyplot as plt
import os,sys,time
import yaml

# tensorflow/gpu start-up configuration
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# %env CUDA_DEVICE_ORDER=PCI_BUS_ID
# %env CUDA_VISIBLE_DEVICES=2
import tensorflow as tf
from tensorflow.python.client import device_lib
from utils import residual_block, downsample_block, upsample_block


def main(params):

    train_io = larcv_threadio()
    train_io_cfg = {'filler_name' : 'TrainIO',
                    'verbosity'   : 0,
                    'filler_cfg'  : 'train_io.cfg'}
    train_io.configure(train_io_cfg)
    train_io.start_manager(params['BATCH_SIZE'])
    train_io.next()


    # Reset the graph:
    tf.reset_default_graph()

    g = tf.Graph()
    # First, get the input placeholders:
    feed_images, feed_labels = init_inputs(params, g, train_io.fetch_data("main_data"), train_io.fetch_data("main_label"))
    labels = tf.squeeze(feed_labels, axis=-1)

    # Apply UResNet to the image to get the logits:
    logits = build_network(params, g, feed_images, training=params['TRAINING'])

    losses = loss(params, labels, logits, g)

    accuracy(params, labels, logits, g)
    global_step, opt = optimizer(params, losses, g)
    snapshot(params, labels, logits, g)
    train_writer, merged_summary = summary(params, g)

    # Perform the training:
    with g.as_default():
        with tf.Session() as sess:
            if not params['RESTORE']:
                sess.run(tf.global_variables_initializer())
                train_writer.add_graph(sess.graph)
                saver = tf.train.Saver()
            else:
                latest_checkpoint = tf.train.latest_checkpoint(LOGDIR+"/checkpoints/")
                print "Restoring model from {}".format(latest_checkpoint)
                saver = tf.train.Saver()
                saver.restore(sess, latest_checkpoint)

            print "Begin training ..."

            for i in xrange(params['TRAINING_ITERATIONS']):
                step = sess.run(global_step)

                # First step, prepare the feed dict:
                train_io.next()
                fd = {feed_images : numpy.reshape(train_io.fetch_data("main_data").data(),
                                             train_io.fetch_data("main_data").dim()),
                      feed_labels : numpy.reshape(train_io.fetch_data("main_label").data(),
                                             train_io.fetch_data("main_label").dim())}

                _, summ = sess.run([opt, merged_summary], feed_dict = fd)


                train_writer.add_summary(summ, step)


                if step != 0 and step % params['SAVE_ITERATION'] == 0:
                    saver.save(
                        sess,
                        LOGDIR+"/checkpoints/save",
                        global_step=step)


                # # train_writer.add_summary(summ, i)
                # # sys.stdout.write('Training in progress @ step %d\n' % (step))
                # if i != 0 and int(10*step) == 10*step:
                #     if int(step) == step:
                #         print 'Training in progress @ step %g, g_loss %g, d_loss %g accuracy %g' % (step, g_l, d_l, acc)


def init_inputs(params, graph, data, label):
    with graph.as_default():
        input_image  = tf.placeholder(tf.float32, data.dim(), name="input_image")
        input_labels = tf.placeholder(tf.int64, label.dim(), name="input_image")

        # Squeeze the last index off of the labels:

        return input_image, input_labels


def build_network(params, graph, input_image, training):
    with graph.as_default():

        x = input_image

        # Initial convolution to get to the correct number of filters:
        x = tf.layers.conv2d_transpose(x, params['N_INITIAL_FILTERS'],
                             kernel_size=[5, 5],
                             strides=[1, 1],
                             padding='same',
                             use_bias=False,
                             trainable=training,
                             name="Conv2DInitial",
                             reuse=None)

        # ReLU:
        x = tf.nn.relu(x)

        # Need to keep track of the outputs of the residual blocks before downsampling, to feed
        # On the upsampling side

        network_filters = []

        # Begin the process of residual blocks and downsampling:
        for i in xrange(params['NETWORK_DEPTH']):

            x = residual_block(x, training,
                               batch_norm=False,
                               name="resblock_down_{0}".format(i))

            network_filters.append(x)
            x = downsample_block(x, training,
                                batch_norm=False,
                                name="downsample_{0}".format(i))


        # At the bottom, do another residual block:
        x = residual_block(x, training, batch_norm=False, name="deepest_block")

        # Come back up the network:
        for i in xrange(params['NETWORK_DEPTH']-1, -1, -1):

            # How many filters to return from upsampling?
            n_filters = network_filters[-1].get_shape().as_list()[-1]
            print "Upsampling to make {} filters".format(n_filters)

            # Upsample:
            x = upsample_block(x, training, batch_norm=False,n_output_filters=n_filters, name="upsample_{}".format(i))


            x = tf.concat([x, network_filters[-1]], axis=-1, name='up_concat_{}'.format(i))

            # Remove the recently concated filters:
            network_filters.pop()

            # Include a bottleneck to reduce the number of filters after upsampling:
            x = tf.layers.conv2d(x,
                                 n_filters,
                                 kernel_size=[1,1],
                                 strides=[1,1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 trainable=training,
                                 name="BottleneckUpsample_{}".format(i))

            x = tf.nn.relu(x)

            # Residual
            x = residual_block(x, training,
                               batch_norm=False,
                               name="resblock_up_{0}".format(i))

        # At this point, we ought to have a network that has the same shape as the initial input, but with more filters.
        # We can use a bottleneck to map it onto the right dimensions:
        x = tf.layers.conv2d(x,
                             params['NUM_LABELS'],
                             kernel_size=[1,1],
                             strides=[1, 1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             trainable=training,
                             name="BottleneckConv2D",)

        # The final activation is softmax across the pixels.  It gets applied in the loss function
#         x = tf.nn.softmax(x)
        return x

def loss(params, labels, logits, graph):
    with graph.as_default(), tf.name_scope('cross_entropy'):

        # For this application, since background pixels heavily outnumber labeled pixels, we can weight
        # the loss to balance things out.
        if params['BALANCE_LOSS']:
            weight  = tf.size(labels) / tf.cast(tf.count_nonzero(labels), tf.int32)
            weights = tf.where(labels==0, x=1, y=weight)
            losses  = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=weights))
        else:
            # Compute the loss without balancing:
            losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels, logits))

        # Add the loss to the summary:
        tf.summary.scalar("Total_Loss", losses)

        return losses

def accuracy(params, labels, logits, graph):
    with graph.as_default(), tf.name_scope('accuracy'):
        total_accuracy   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1),
                                                           labels), tf.float32))
        predicted_labels = tf.argmax(logits, axis=-1)
        # Find the non zero labels:
        non_zero_indices = tf.not_equal(labels, tf.constant(0, labels.dtype))

        non_zero_logits = tf.boolean_mask(predicted_labels, non_zero_indices)
        non_zero_labels = tf.boolean_mask(labels, non_zero_indices)

        non_bkg_accuracy = tf.reduce_mean(tf.cast(tf.equal(non_zero_logits, non_zero_labels), tf.float32))

        # Add the accuracies to the summary:
        tf.summary.scalar("Total_Accuracy", total_accuracy)
        tf.summary.scalar("Non_Background_Accuracy", non_bkg_accuracy)

def optimizer(params, loss, graph):
    with graph.as_default(), tf.name_scope('training'):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        opt = tf.train.AdamOptimizer(params['BASE_LEARNING_RATE']).minimize(loss, global_step = global_step)
        return global_step, opt

def snapshot(params, labels, logits, graph):
    with graph.as_default(), tf.name_scope('snapshot'):
        # There are 5 labels plus background pixels available:
        predicted_label = tf.argmax(logits, axis=-1)

        print predicted_label.get_shape()

        for label in xrange(len(params['LABEL_NAMES'])):
            target_img = tf.cast(tf.equal(labels, tf.constant(label, labels.dtype)), tf.float32)
            output_img = tf.cast(tf.equal(predicted_label, tf.constant(label, labels.dtype)), tf.float32)
            tf.summary.image('{}_labels'.format(params['LABEL_NAMES'][label]), tf.reshape(target_img, target_img.get_shape().as_list() + [1,]))
            tf.summary.image('{}_logits'.format(params['LABEL_NAMES'][label]), tf.reshape(output_img, output_img.get_shape().as_list() + [1,]))


def summary(params, graph):
    with graph.as_default():
        merged_summary = tf.summary.merge_all()

        # Set up a saver:
        train_writer = tf.summary.FileWriter(params['LOGDIR'])
        return train_writer, merged_summary


if __name__ == '__main__':
    with open('train.yml', 'r') as f:
        params = yaml.load(f)
    main(params)

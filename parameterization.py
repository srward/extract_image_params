#!/usr/bin/python
activate_this_file = "/course/cs1430/tf_gpu/bin/activate_this.py"
execfile(activate_this_file, dict(__file__=activate_this_file))

import tensorflow as tf
import numpy as np
import os
import Image
from glob import glob
import scipy.misc
import random


BATCH_SZ = 32
num_epochs = 1000
learning_rate = 0.01

def preprocessNoConcat(image_dir):
    files = glob('./datasets/{}/*.*'.format(image_dir))

    imgs = []
    for i in range(len(files)):
        img = scipy.misc.imread(files[i], mode='RGB').astype(np.float)
        imgs.append(img)
    # print(np.shape(np.array(imgs)))

    return imgs


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def create_batch(train_data, t):
    X = np.zeros((BATCH_SZ, 256, 256, 3), dtype=np.float32)
        
    for k, image in enumerate(train_data[t:t+BATCH_SZ]):
        X[k, :, :, :] = image / 255.0
        
    return X

def save_images(images, image_path):
    return scipy.misc.imsave(image_path, images)



###################################################
#               MODEL                             #
###################################################

def c_cnn(images):

  conv1 = tf.layers.conv2d(
      inputs= images,
      filters=16,
      kernel_size=[25, 25])


  conv_out = batch_norm(conv1, "bn1")
  conv_out = lrelu(conv_out, 0.2, "relu1")
  conv_out = tf.layers.max_pooling2d(conv_out, pool_size=[2, 2], strides=2)


  conv2 = tf.layers.conv2d(
      inputs=conv_out,
      filters= 32,
      kernel_size=[25, 25])

  conv_out = batch_norm(conv2, "bn2")
  conv_out = lrelu(conv_out, 0.2, "relu2")

  conv3 = tf.layers.conv2d(
      inputs=conv_out,
      filters=64,
      kernel_size=[25, 25])

  conv_out = batch_norm(conv3, "bn3")
  conv_out = lrelu(conv_out, 0.2, "relu3")


  conv4 = tf.layers.conv2d(
      inputs=conv_out,
      filters=128,
      kernel_size=[25, 25])

  conv_out = batch_norm(conv4, "bn4")
  conv_out = lrelu(conv_out, 0.2, "relu4")

  conv5 = tf.layers.conv2d(
       inputs=conv_out,
       filters=256,
       kernel_size=[25, 25])

  conv_out = batch_norm(conv5, "bn5")
  conv_out = lrelu(conv_out, 0.2, "relu5")

  conv6 = tf.layers.conv2d(
       inputs=conv_out,
       filters=512,
       kernel_size=[18, 20])

  conv_out = batch_norm(conv6, "bn6")
  conv_out = lrelu(conv_out, 0.2, "relu6")

  return conv_out

def e_cnn(parameters):

  conv1 = tf.layers.conv2d_transpose(
          inputs= parameters,
          filters=512,
          kernel_size=[12, 14],
          name="conv1-e")

  conv_bn = batch_norm(conv1, name="batch_norm1-e")
  conv_out = tf.nn.relu(conv1, "relu1-e")


  conv2 = tf.layers.conv2d_transpose(
          inputs=conv_out,
          filters=256,
          kernel_size=[4, 4],
          strides=(2,2),
          name="conv2-e")

  conv_bn = batch_norm(conv2, name="batch_norm2-e")
  conv_out = tf.nn.relu(conv_bn, "relu2-e")


  conv3 = tf.layers.conv2d_transpose(
          inputs=conv_out,
          filters=128,
          kernel_size=[4, 4],
          strides=(2,2),
          name="conv3-e")

  conv_bn = batch_norm(conv3, name="batch_norm3-e")
  conv_out = tf.nn.relu(conv_bn, "relu3-e")


  conv4 = tf.layers.conv2d_transpose(
          inputs=conv_out,
          filters=64,
          kernel_size=[4, 4],
          strides=(2,2),
          name="conv4-e")


  conv_bn = batch_norm(conv4, name="batch_norm4-e")
  conv_out = tf.nn.relu(conv_bn, "relu4-e")

  conv5 = tf.layers.conv2d_transpose(
          inputs=conv_out,
          filters=3,
          kernel_size=[6, 6],
          strides=(2,2),
          name="conv5-e")


  return conv5



images = tf.placeholder(tf.float32, [None, 256, 256, 3])

c_out = c_cnn(images)
conv_final = tf.tanh(c_out)

# flattened parameters
flattened = tf.contrib.layers.flatten(conv_final)
# fully connected layer to get to a size of 3
parameters = tf.layers.dense(flattened, 3)
reshaped = tf.reshape(parameters, [-1, 3, 1, 1])

conv_up = e_cnn(reshaped)
regularized = tf.sigmoid(conv_up)


labels = tf.placeholder(tf.float32, [None, 256, 256, 3])

loss = tf.reduce_mean(tf.square(tf.subtract(regularized, labels)))
# loss = tf.losses.absolute_difference(conv_final, labels, 1.0, None, tf.GraphKeys.LOSSES, tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
# loss = tf.reduce_mean(tf.square(tf.subtract(c_out, labels)))

global_step = tf.Variable(0, trainable=False)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


#########################################
#              TRAIN                    #
#########################################

train_data = preprocessNoConcat('trainEdited')

session = tf.Session()
session.run(tf.global_variables_initializer())

for x in range(num_epochs):
  # for i in range(1):
  for i in range(0, np.array(train_data).shape[0], BATCH_SZ):
    print("Epoch: [" + str(x) + "]     Training step: [" + str((i/32)) + "]")

    next_image = create_batch(train_data, i)

    # Collect/Run Loss, Training Operation via single call to session.run (note multiple fetches!)
    l, _, output_img = session.run([loss, train_op, regularized], feed_dict={images: next_image, labels: next_image})
    print 'loss = %s' % l



########################################
#            TESTING                   #
########################################

test_dir = "testRes1"

files = glob('CycleGAN-output/*.jpg')
param_file = open(os.path.join(test_dir, "param_file"), "w+")
num_tests = 100
for i in range(num_tests):
  print("Test step: " + str(i))
  next_image = scipy.misc.imread(files[i], mode='RGB').astype(np.float)
  next_image = np.expand_dims(next_image, axis=0)
  next_image = next_image/255.0


  parameters, output_img = session.run([reshaped, regularized], feed_dict={images: next_image, labels: next_image})

  parameters = np.squeeze(parameters)
  output_img = np.squeeze(output_img, axis=0)
  output_img = output_img * 255

  param_file.writelines(["%s, " % item  for item in parameters])
  param_file.write('\n')
  image_path = os.path.join(test_dir,
                            '{0}'.format(os.path.basename(files[i])))

  save_images(output_img, image_path)


param_file.close()
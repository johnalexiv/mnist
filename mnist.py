import tensorflow as tf
import os


LOGDIR = "/Users/john/Documents/MNIST/"
# LABELS = os.path.join(os.getcwd(), "labels_1024.tsv")
# SPRITES = os.path.join(os.getcwd(), "sprite_1024.png")

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)

# if not (os.path.isfile(LABELS) and os.path.isfile(SPRITES)):
#   print("Necessary data files were not found. Run this command from inside the "
#     "repo provided at "
#     "https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial.")
#   exit(1)

def weight_variable(shape):
    weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weight)

def bias_variable(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)

def neural_network(input, num_input, num_layers, num_neurons, num_classes):

    input_channel = num_input
    output_channel = num_neurons
    prev_layer = input

    if num_layers > 1:
        for _ in range(num_layers):
            out_layer = tf.add(tf.matmul(
                prev_layer, weight_variable([input_channel, output_channel])),
                bias_variable([output_channel]))

            prev_layer = out_layer
            input_channel = output_channel

    final_layer = tf.add(tf.matmul(
        prev_layer, weight_variable([input_channel, num_classes])),
        bias_variable([num_classes]))

    return final_layer

def mnist_model(learning_rate,
                batch_size,
                num_epochs,
                display_step,
                num_neurons,
                num_layers,
                num_input,
                num_classes,
                hparam):

    # Reset the graph
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, num_input])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

    # Output of network
    y = neural_network(x, num_input, num_layers, num_neurons, num_classes)

    # Cost function
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('Train'):
        # Gradient descent optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # Back propagation step
        train_step = optimizer.minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        # Get prediction and calculate accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # Merge summaries
    summary = tf.summary.merge_all()

    # Setup saver to save variables and graph
    saver = tf.train.Saver()

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)

    # Train
    for step in range(num_epochs):
        batch = mnist.train.next_batch(batch_size)

        # Display accuracy
        if step % display_step == 0:
            [train_accuracy, sum] = sess.run([accuracy, summary], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print('training accuracy: %s, step: %d'%(train_accuracy, step))
            writer.add_summary(sum, step)

        # Training step
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

def make_hparam_string(learning_rate, num_neurons, num_layers):
    return "lr_%.0E,neurons=%s,layers=%s" % (learning_rate, num_neurons, num_layers)

def main():
    # Parameters
    learning_rate = 0.1
    batch_size = 50
    num_epochs = 5000
    display_step = 100

    # Network Parameters
    num_neurons = 50  # number of neurons in each layer
    num_layers = 2  # number of layers
    num_input = 784  # MNIST data input (img shape: 28*28)
    num_classes = 10  # MNIST total classes (0-9 digits)

    hparam = make_hparam_string(learning_rate, num_neurons, num_layers)
    print('Starting run for %s'%hparam)

    mnist_model(learning_rate,
                batch_size,
                num_epochs,
                display_step,
                num_neurons,
                num_layers,
                num_input,
                num_classes,
                hparam)

    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
    print('Running on mac? If you want to get rid of the dialogue asking to give '
          'network permissions to TensorBoard, you can provide this flag: '
          '--host=localhost')

if __name__ == '__main__':
    main()
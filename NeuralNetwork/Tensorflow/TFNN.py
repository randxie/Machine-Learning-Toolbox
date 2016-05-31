__author__ = 'randxie'

import tensorflow as tf
import numpy as np
class TFNN():
    def __init__(self):
        self.learning_rate = 0.0025
        self.training_epochs = 300
        self.batch_size = 50
        self.display_step = 100
        self.num_save_record = 2
        self.l2reg = 1e-5
        self.keep_prob = 0.5 # Dropout, probability to keep units
        self.epoch_per_save = int(self.training_epochs/self.num_save_record)
        # Network Parameters
        self.n_hidden_1 = 20 # 1st layer num features
        self.n_hidden_2 = 20 # 2nd layer num features

    def init_struct(self, data_in):
        self.n_input = data_in.shape[1] 
        self.n_classes = 2 

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        #self.y = tf.nn.softmax(tf.matmul(self.x,self.weights) + self.biases)
        self.y = tf.placeholder("float", [None, self.n_classes])

        # define network structure
        self.mlp_net = self.multilayer_perceptron()

        # define loss
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.mlp_net, self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.init = tf.initialize_all_variables()

        # define saver
        self.saver = tf.train.Saver(max_to_keep=self.num_save_record)

    def fit(self, data_in, data_out):
        self.init_struct(data_in)
        self.sess = tf.Session()
        self.sess.run(self.init)

        # Training cycle
        for epoch in range(self.training_epochs):
            avg_cost = 0.
            num_datap = data_in.shape[0]
            total_batch = int(data_in.shape[0]/self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                if ((i+1)*self.batch_size < num_datap):
                    batch_xs = data_in[i*self.batch_size:(i+1)*self.batch_size,:]
                    batch_ys = data_out[i*self.batch_size:(i+1)*self.batch_size]
                else:
                    batch_xs = data_in[i*self.batch_size:,:]
                    batch_ys = data_out[i*self.batch_size:]
                # Fit training using batch data
                batch_ys = np.vstack((1-batch_ys, batch_ys)).T
                self.sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys, self.tf_keep_prob: self.keep_prob})
                # Compute average loss
                avg_cost += self.sess.run(self.cost, feed_dict={self.x: batch_xs, self.y: batch_ys, self.tf_keep_prob: self.keep_prob})/total_batch
            if ((1 + epoch) % (self.epoch_per_save) == 0):
                self.saver.save(self.sess, 'my_model_tf',global_step=epoch)
            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
                print self.sess.run(self.weights['h1'])
                
        print "Optimization Finished!"
        '''
        correct_prediction = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print 'accuracy:'
        print(self.sess.run(accuracy, feed_dict={self.x: data_in, self.y: data_out}))
        '''

    def predict(self, data_in):
        prediction = tf.argmax(self.mlp_net, 1)
        ypred = np.array(prediction.eval(feed_dict={self.x: data_in, self.tf_keep_prob: self.keep_prob}, session=self.sess))
        return ypred

    def predict_proba(self, data_in):
        #pred_test = self.mlp_net(data_in, self.weights, self.biases)
        #ypred = np.array(pred_test.eval())
        #prediction = tf.argmax(self.pred,1)
        prediction = self.mlp_net
        #prediction = tf.nn.softmax(self.mlp_net)
        ypred = np.array(prediction.eval(feed_dict={self.x: data_in, self.tf_keep_prob: self.keep_prob}, session=self.sess))
        print ypred
        ymin = np.amin(ypred, axis=1) 
        ymax = np.amax(ypred, axis=1) 
        yrange = np.subtract(ymax,ymin)
        ypred[:,0] = np.divide((ypred[:,0]-ymin), yrange)
        ypred[:,1] = np.divide((ypred[:,1]-ymin), yrange)
        return ypred

    def multilayer_perceptron(self):
        self.__init_wb__()
        #self.keep_prob = tf.placeholder(tf.float32)
        #layer_0 = tf.nn.dropout(self.x, self.keep_prob)
        #Hidden layer with RELU activation
        layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1']))

        self.tf_keep_prob = tf.placeholder(tf.float32)
        
        layer_1 = tf.nn.dropout(layer_1, self.tf_keep_prob)
        #Hidden layer with RELU activation
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        layer_2 = tf.nn.dropout(layer_2, self.tf_keep_prob)
        return tf.matmul(layer_2, self.weights['out']) + self.biases['out']

    def __init_wb__(self):
        self.weights = {
        'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
        'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

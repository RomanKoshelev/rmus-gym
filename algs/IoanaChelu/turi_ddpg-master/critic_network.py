import tensorflow as tf
import config
import math


class CriticNetwork:
    def __init__(self, state_size, action_size):
        with tf.Graph().as_default():
            self.sess = tf.InteractiveSession()

            # critic network params:
            self.w1, self.b1, self.w2_q, self.w2_action, self.b2, self.w3, self.b3, \
              self.state_input, self.action_input, self.q_output = self.create_network(state_size, action_size)

            # target critic network params:
            self.t_w1, self.t_b1, self.t_w2_q, self.t_w2_action, self.t_b2, self.t_w3, self.t_b3, \
              self.t_state_input, self.t_action_input, self.t_q_output = self.create_network(state_size, action_size)

            # cost network
            self.y_input = tf.placeholder("float", [None, 1])
            self.l2_regularizer_loss = config.L2_DECAY * tf.reduce_sum(
                tf.pow(self.w2_q, 2)) + config.L2_DECAY * tf.reduce_sum(
                tf.pow(self.b2, 2))
            self.cost = tf.pow(self.q_output - self.y_input, 2) / config.MINI_BATCH_SIZE + self.l2_regularizer_loss
            self.optimizer = tf.train.AdamOptimizer(config.LEARN_RATE).minimize(self.cost)

            # action gradient to be used in actor network:
            self.action_grad = tf.gradients(self.q_output, self.action_input)
            self.action_grad = self.action_grad[0] / config.MINI_BATCH_SIZE

            self.sess.run(tf.initialize_all_variables())
            self.saver = tf.train.Saver()

            checkpoint = tf.train.get_checkpoint_state(config.CHECKPOINT_PATH_CRITIC)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print "Successfully loaded:", checkpoint.model_checkpoint_path
            else:
                print "Could not find old network weights"

            # To make sure actor and target have same intial parmameters copy the parameters
            # copy target parameters
            self.sess.run([
                self.t_w1.assign(self.w1),
                self.t_b1.assign(self.b1),
                self.t_w2_q.assign(self.w2_q),
                self.t_w2_action.assign(self.w2_action),
                self.t_b2.assign(self.b2),
                self.t_w3.assign(self.w3),
                self.t_b3.assign(self.b3)
            ])

    def get_target_q_batch(self, state_batch, action_batch):
        return self.sess.run(self.t_q_output,
                             feed_dict={self.t_state_input: state_batch, self.t_action_input: action_batch})

    def create_network(self, state_size, action_size):
        state_input = tf.placeholder("float", [None, state_size])
        action_input = tf.placeholder("float", [None, action_size])

        w1 = self.random_uniform_variable([state_size, config.HIDDEN_LAYER_1], state_size)
        b1 = self.random_uniform_variable([config.HIDDEN_LAYER_1], state_size)
        w2_q = self.random_uniform_variable([config.HIDDEN_LAYER_1, config.HIDDEN_LAYER_2],
                                            config.HIDDEN_LAYER_1 + action_size)
        w2_action = self.random_uniform_variable([action_size, config.HIDDEN_LAYER_2],
                                                 config.HIDDEN_LAYER_1 + action_size)
        b2 = self.random_uniform_variable([config.HIDDEN_LAYER_2], config.HIDDEN_LAYER_1 + action_size)
        w3 = self.random_uniform_variable([config.HIDDEN_LAYER_2, action_size], 0.0003)
        b3 = self.random_uniform_variable([action_size], 0.0003)

        hidden1 = tf.nn.relu(tf.matmul(state_input, w1) + b1)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, w2_q) + tf.matmul(action_input, w2_action) + b2)
        output = tf.matmul(hidden2, w3) + b3

        return w1, b1, w2_q, w2_action, b2, w3, b3, state_input, action_input, output

    def random_uniform_variable(self, shape, fan_in):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in)))

    def save_network(self, time_step):
        self.saver.save(self.sess, config.CHECKPOINT_PATH_CRITIC, global_step=time_step)

    def train(self, y_input_batch, state_batch, action_batch):
        self.sess.run(self.optimizer, feed_dict={self.y_input: y_input_batch, self.action_input: action_batch,
                                                 self.state_input: state_batch})

    def get_gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_grad,
                             feed_dict={self.action_input: action_batch, self.state_input: state_batch})

    def update_target(self):
        self.sess.run([
            self.t_w1.assign((1 - config.TAO) * self.t_w1 + config.TAO * self.w1),
            self.t_b1.assign((1 - config.TAO) * self.t_b1 + config.TAO * self.b1),
            self.t_w2_q.assign((1 - config.TAO) * self.t_w2_q + config.TAO * self.w2_q),
            self.t_w2_action.assign((1 - config.TAO) * self.t_w2_action + config.TAO * self.w2_action),
            self.t_b2.assign((1 - config.TAO) * self.t_b2 + config.TAO * self.b2),
            self.t_w3.assign((1 - config.TAO) * self.t_w3 + config.TAO * self.w3),
            self.t_b3.assign((1 - config.TAO) * self.t_b3 + config.TAO * self.b3),
        ])

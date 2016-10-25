import tensorflow as tf
import config
import math


class ActorNetwork:
    def __init__(self, state_size, action_size):
        with tf.Graph().as_default():
            self.sess = tf.InteractiveSession()

            # actor network params:
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, \
            self.state_input, self.action_output = self.create_network(state_size, action_size)

            # target actor network params:
            self.t_w1, self.t_b1, self.t_w2, self.t_b2, self.t_w3, self.t_b3, \
            self.t_state_input, self.t_action_output = self.create_network(state_size, action_size)

            # cost network
            # gets input from action_gradient computed in critic network
            self.q_gradient_input = tf.placeholder("float", [None, action_size])

            self.actor_params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
            self.params_gradients = tf.gradients(self.action_output, self.actor_params, -self.q_gradient_input)

            self.optimizer = tf.train.AdamOptimizer(config.LEARN_RATE).apply_gradients(
                zip(self.params_gradients, self.actor_params))

            self.sess.run(tf.initialize_all_variables())

            self.saver = tf.train.Saver()

            checkpoint = tf.train.get_checkpoint_state(config.CHECKPOINT_PATH_ACTOR)
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
                self.t_w2.assign(self.w2),
                self.t_b2.assign(self.b2),
                self.t_w3.assign(self.w3),
                self.t_b3.assign(self.b3)
            ])

    def get_action(self, state):
        return self.sess.run(self.action_output, feed_dict={self.state_input: [state]})[0]

    def get_target_action_batch(self, state_batch):
        return self.sess.run(self.t_action_output, feed_dict={self.t_state_input: state_batch})

    def get_action_batch(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={self.state_input: state_batch})

    def create_network(self, state_size, action_size):
        state_input = tf.placeholder("float", [None, state_size])

        w1 = self.random_uniform_variable([state_size, config.HIDDEN_LAYER_1], state_size)
        b1 = self.random_uniform_variable([config.HIDDEN_LAYER_1], state_size)
        w2 = self.random_uniform_variable([config.HIDDEN_LAYER_1, config.HIDDEN_LAYER_2], config.HIDDEN_LAYER_1)
        b2 = self.random_uniform_variable([config.HIDDEN_LAYER_2], config.HIDDEN_LAYER_1)
        w3 = self.random_uniform_variable([config.HIDDEN_LAYER_2, action_size], 0.0003)
        b3 = self.random_uniform_variable([action_size], 0.0003)

        hidden1 = tf.nn.relu(tf.matmul(state_input, w1) + b1)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)
        output = tf.matmul(hidden2, w3) + b3

        return w1, b1, w2, b2, w3, b3, state_input, output

    def random_uniform_variable(self, shape, fan_in):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in)))

    def save_network(self, time_step):
        self.saver.save(self.sess, config.CHECKPOINT_PATH_ACTOR, global_step=time_step)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer,
                      feed_dict={self.q_gradient_input: q_gradient_batch, self.state_input: state_batch})

    def update_target(self):
        self.sess.run([
            self.t_w1.assign((1 - config.TAO) * self.t_w1 + config.TAO * self.w1),
            self.t_b1.assign((1 - config.TAO) * self.t_b1 + config.TAO * self.b1),
            self.t_w2.assign((1 - config.TAO) * self.t_w2 + config.TAO * self.w2),
            self.t_b2.assign((1 - config.TAO) * self.t_b2 + config.TAO * self.b2),
            self.t_w3.assign((1 - config.TAO) * self.t_w3 + config.TAO * self.w3),
            self.t_b3.assign((1 - config.TAO) * self.t_b3 + config.TAO * self.b3),
        ])

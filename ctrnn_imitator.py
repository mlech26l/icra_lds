import tensorflow as tf
import os
import numpy as np


def affine_layer(inputs,activation=None,name=""):
    input_size = int(inputs.shape[-1])
    k = tf.get_variable('affine_k_{}'.format(name), [input_size],initializer=tf.initializers.constant(1.0))
    d = tf.get_variable("affine_d_{}".format(name), [input_size],initializer=tf.initializers.constant(0.0))

    y = inputs*k + d
    if(not activation is None):
        y = activation(y)

    return y

class CTRNN_Cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units,cell_clip=-1):
        self._num_units = num_units
        self._num_unfolds = 6
        self._delta_t = 1.0/self._num_unfolds

        self.tau = 1.0
        self.cell_clip = cell_clip


    def _dense(self,units,inputs,activation,name,bias_initializer=tf.constant_initializer(0.0)):
        input_size = int(inputs.shape[-1])
        W = tf.get_variable('W_{}'.format(name), [input_size, units])
        b = tf.get_variable("bias_{}".format(name), [units],initializer=bias_initializer)

        y = tf.matmul(inputs,W) + b
        if(not activation is None):
            y = activation(y)

        return y

    def linear(self,units,inputs,name=""):
        input_size = int(inputs.shape[-1])
        W = tf.get_variable('W_linear'+name, [input_size, units])

        y = tf.matmul(inputs,W)

        return y,W


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self,input_shape):
        pass


    def __call__(self, inputs, state, scope=None):
        # CTRNN ODE is: df/dt = NN(x) - f
        # where x is the input, and NN is a MLP.
        # Input could be: 1: just the input of the RNN cell
        # or 2: input of the RNN cell merged with the current state

        self._input_size = int(inputs.shape[-1])
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("RNN",reuse=tf.AUTO_REUSE):  

                for i in range(self._num_unfolds):
                    fused_input = tf.concat([state,inputs],axis=-1)
                    input_f_prime = self._dense(inputs=fused_input,units=self._num_units,activation=tf.nn.tanh,name="l")
                    # df/dt 
                    f_prime = -state/self.tau + input_f_prime
                    # If we solve this ODE with explicit euler we get
                    # f(t+deltaT) = f(t) + deltaT * df/dt
                    state = state + self._delta_t * f_prime

                    if(self.cell_clip > 0):
                        state = tf.clip_by_value(state,-self.cell_clip,self.cell_clip)

        return state,state

class CTRNN_Imitator:

    def __init__(self,num_units,clip_value,lidar_size,state_size):

        self.lidar_size = lidar_size
        self.state_size = state_size

        self.num_units = num_units

        self.x_lidar = tf.placeholder(tf.float32, shape=[None, None, self.lidar_size],name='lidar')
        self.x_state = tf.placeholder(tf.float32, shape=[None, None, self.state_size],name='state')
        self.target_y = tf.placeholder(tf.float32, shape=[None,None,1])

        head = tf.reshape(self.x_lidar,[-1,self.lidar_size,1])
        head = tf.keras.layers.Conv1D(12,5,strides=3,activation="relu")(head)
        head = tf.keras.layers.Conv1D(16,5,strides=3,activation="relu")(head)
        head = tf.keras.layers.Conv1D(24,5,strides=2,activation="relu")(head)
        head = tf.keras.layers.Conv1D(1,1,strides=1,activation=None)(head)
        head = tf.keras.layers.Flatten()(head)

        head = tf.reshape(head,shape=[tf.shape(self.x_lidar)[0],tf.shape(self.x_lidar)[1],head.shape[-1]])

        estim_in = self.x_state
        estim_in = tf.clip_by_value(estim_in,-1.0,1.0)
        head = tf.concat([head,estim_in],axis=-1)
        print("head shape: ",str(head.shape))

        self.init_state = tf.placeholder(tf.float32,[None,self.num_units],name="initial_state")

        self.fused_cell = CTRNN_Cell(self.num_units,cell_clip=clip_value)

        cell_out,self.final_state = tf.nn.dynamic_rnn(self.fused_cell,head,initial_state = self.init_state,time_major=True)

        cell_out = tf.reshape(cell_out,[-1,self.num_units])

        y = tf.keras.layers.Dense(1,activation=None,name="ct_out")(cell_out)
        # Reshape back to sequenced batch form
        self.y = tf.reshape(y,shape=[tf.shape(self.x_state)[0],tf.shape(self.x_state)[1],1])

        #Output
        tf.identity(self.y,name='prediction')
        tf.identity(self.final_state,name='final_state')

        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target_y, self.y)))

        # Loss, error and training algorithm
        self.mean_abs_error = tf.reduce_mean(tf.abs(tf.subtract(self.y, self.target_y)))

        optimizer = tf.train.AdamOptimizer(0.0001)
        self.train_step = optimizer.minimize(self.loss)

    def zero_state(self,batch_size):
        return np.zeros([batch_size,self.num_units],dtype=np.float32)

    def share_sess(self, sess):
        self.sess = sess


    def predict_step(self, x_state, x_lidar,init_state=None):
        if(init_state is None):
            init_state = self.zero_state(1)

        # Reshape sequence into a batch of 1 sequence
        x_state = x_state.reshape([1,1,self.state_size])
        x_lidar = x_lidar.reshape([1,1,self.lidar_size])
        feed_dict = {
            self.x_state: x_state,
            self.x_lidar: x_lidar,
            self.init_state: init_state}

        prediction,next_state = self.sess.run([self.y,self.final_state], feed_dict=feed_dict)
        return float(prediction.flatten()),next_state

    
    def evaluate(self, batch_x_state, batch_x_lidar,batch_y):
        feed_dict = {
            self.x_state: batch_x_state,
            self.x_lidar: batch_x_lidar,
            self.target_y: batch_y,
            self.init_state:self.zero_state(batch_x_state.shape[1])
            }

        loss,mae = self.sess.run([self.loss,self.mean_abs_error], feed_dict=feed_dict)

        return loss,mae
    def train_iter(self, batch_x_state, batch_x_lidar,batch_y):
        feed_dict = {
            self.x_state: batch_x_state,
            self.x_lidar: batch_x_lidar,
            self.target_y: batch_y,
            self.init_state:self.zero_state(batch_x_state.shape[1])
            }

        (_,loss,mae) = self.sess.run([self.train_step, self.loss,self.mean_abs_error], feed_dict=feed_dict)

        return loss,mae

    def create_checkpoint(self, path, name='model'):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, '-'+name)
        self.saver = tf.train.Saver()
        filename = self.saver.save(self.sess, checkpoint_path)

    def restore_from_checkpoint(self, path):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(path,'-model'))


class CTRNN_Cheetah:

    def __init__(self,num_units,clip_value,obs_size,action_size):

        self.obs_size = obs_size
        self.action_size = action_size

        self.num_units = num_units

        self.x_obs = tf.placeholder(tf.float32, shape=[None, None, self.obs_size],name='state')
        self.target_y = tf.placeholder(tf.float32, shape=[None,None,self.action_size])

        head = self.x_obs
        head = tf.reshape(self.x_obs,[-1,self.obs_size])
        head = tf.keras.layers.Dense(128,activation="relu")(head)
        head = tf.keras.layers.Dense(128,activation=None)(head)
        head = tf.reshape(head,shape=[tf.shape(self.x_obs)[0],tf.shape(self.x_obs)[1],head.shape[-1]])

        self.init_state = tf.placeholder(tf.float32,[None,self.num_units],name="initial_state")

        self.fused_cell = CTRNN_Cell(self.num_units,cell_clip=clip_value)

        cell_out,self.final_state = tf.nn.dynamic_rnn(self.fused_cell,head,initial_state = self.init_state,time_major=True)

        cell_out = tf.reshape(cell_out,[-1,self.num_units])

        y = tf.keras.layers.Dense(self.action_size,activation=None,name="ct_out")(cell_out)
        # Reshape back to sequenced batch form
        self.y = tf.reshape(y,shape=[tf.shape(self.x_obs)[0],tf.shape(self.x_obs)[1],self.action_size])

        #Output
        tf.identity(self.y,name='prediction')
        tf.identity(self.final_state,name='final_state')

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(self.target_y, self.y)),axis=-1))

        # Loss, error and training algorithm
        self.mean_abs_error = tf.reduce_mean(tf.abs(tf.subtract(self.y, self.target_y)))

        optimizer = tf.train.AdamOptimizer(0.0001)
        self.train_step = optimizer.minimize(self.loss)

    def zero_state(self,batch_size):
        return np.zeros([batch_size,self.num_units],dtype=np.float32)

    def share_sess(self, sess):
        self.sess = sess


    def predict_step(self, x_obs,init_state=None):
        if(init_state is None):
            init_state = self.zero_state(1)

        # Reshape sequence into a batch of 1 sequence
        x_obs = x_obs.reshape([1,1,self.obs_size])
        feed_dict = {
            self.x_obs: x_obs,
            self.init_state: init_state}

        prediction,next_state = self.sess.run([self.y,self.final_state], feed_dict=feed_dict)
        return prediction.flatten(),next_state

    
    def evaluate(self, batch_x_obs, batch_action):
        feed_dict = {
            self.x_obs: batch_x_obs,
            self.target_y: batch_action,
            self.init_state:self.zero_state(batch_x_obs.shape[1])
            }

        loss,mae = self.sess.run([self.loss,self.mean_abs_error], feed_dict=feed_dict)

        return loss,mae
    def train_iter(self,  batch_x_obs, batch_action):
        feed_dict = {
            self.x_obs: batch_x_obs,
            self.target_y: batch_action,
            self.init_state:self.zero_state(batch_x_obs.shape[1])
            }

        (_,loss,mae) = self.sess.run([self.train_step, self.loss,self.mean_abs_error], feed_dict=feed_dict)


        return loss,mae

    def create_checkpoint(self, path, name='model'):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, '-'+name)
        self.saver = tf.train.Saver()
        filename = self.saver.save(self.sess, checkpoint_path)

    def restore_from_checkpoint(self, path):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(path,'-model'))

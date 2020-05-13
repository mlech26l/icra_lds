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

class LDS_Cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units,cell_clip=-1):
        self._num_units = num_units
        self._num_unfolds = 6
        self._delta_t = 1.0/self._num_unfolds

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
        W = tf.get_variable('W_linear'+name, [input_size, units],initializer=tf.truncated_normal_initializer(stddev=0.01))

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

        self._input_size = int(inputs.shape[-1])
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("RNN",reuse=tf.AUTO_REUSE):  # Reset gate and update gate.

                for i in range(self._num_unfolds):
                    state_f_prime,self.W = self.linear(inputs=state,units=self._num_units)
                    in_f_prime,b = self.linear(inputs=inputs,units=self._num_units,name="inputs")

                    f_prime = state_f_prime + in_f_prime
                    # df/dt 

                    # If we solve this ODE with explicit euler we get
                    # f(t+deltaT) = f(t) + deltaT * df/dt
                    state = state + self._delta_t * f_prime

        return state,state

class LDS_Imitator:

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
        head = affine_layer(head,name="input")

        self.init_state = tf.placeholder(tf.float32,[None,self.num_units],name="initial_state")

        self.fused_cell = LDS_Cell(self.num_units,cell_clip=clip_value)

        cell_out,self.final_state = tf.nn.dynamic_rnn(self.fused_cell,head,initial_state = self.init_state,time_major=True)

        cell_out = tf.reshape(cell_out,[-1,self.num_units])

        cell_out = affine_layer(cell_out)
        
        y = tf.keras.layers.Dense(1,activation=None,name="ct_out")(cell_out)
        # Reshape back to sequenced batch form
        self.y = tf.reshape(y,shape=[tf.shape(self.x_state)[0],tf.shape(self.x_state)[1],1])

        #Output
        tf.identity(self.y,name='prediction')
        tf.identity(self.final_state,name='final_state')

        self.add_stablity_optimization()

        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target_y, self.y))) + 0.0*self.surrogate_loss

        # Loss, error and training algorithm
        self.mean_abs_error = tf.reduce_mean(tf.abs(tf.subtract(self.y, self.target_y)))

        optimizer = tf.train.AdamOptimizer(0.0001)
        self.train_step = optimizer.minimize(self.loss)

    def zero_state(self,batch_size):
        return np.zeros([batch_size,self.num_units],dtype=np.float32)

    def share_sess(self, sess):
        self.sess = sess

    def is_stable(self):
        A = self.sess.run(self.fused_cell.W)
        e,v = np.linalg.eig(A)
        stable = True
        for i in range(e.shape[0]):
            r = np.real(e[i])
            if(r > 0.0):
                stable = False
        return stable

    def add_stablity_optimization(self):
        diag_part = tf.diag_part(self.fused_cell.W)
        row_reduced = tf.reduce_sum(tf.abs(self.fused_cell.W),axis=1)
        abs_diag = tf.abs(diag_part)

        # Summation trick to avoid summing over the diagonal
        R = row_reduced-abs_diag

        # Upper bound on lambda on real axis
        lambda_ub = diag_part + R

        self.surrogate_loss = tf.reduce_sum(tf.nn.relu(lambda_ub))
        optimizer = tf.train.AdamOptimizer(0.0001)
        self.opt_stability = optimizer.minimize(self.surrogate_loss)



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

        w_path = os.path.join(path,"w.csv")
        A = self.sess.run(self.fused_cell.W)

        np.savetxt(w_path,A)

        e,v = np.linalg.eig(A)
        w_info = os.path.join(path,"eigen.txt")
        with open(w_info,"w") as f:
            f.write("Eigenvalues: "+str(e))
        

    def restore_from_checkpoint(self, path):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(path,'-model'))




class LDS_Cheetah:

    def __init__(self,num_units,clip_value,obs_size,action_size):

        self.obs_size = obs_size
        self.action_size = action_size

        self.num_units = num_units

        self.x_obs = tf.placeholder(tf.float32, shape=[None, None, self.obs_size],name='state')
        self.target_y = tf.placeholder(tf.float32, shape=[None,None,self.action_size])
        head = tf.reshape(self.x_obs,[-1,self.obs_size])
        head = tf.keras.layers.Dense(128,activation="relu")(head)
        head = tf.keras.layers.Dense(128,activation=None)(head)
        head = tf.reshape(head,shape=[tf.shape(self.x_obs)[0],tf.shape(self.x_obs)[1],head.shape[-1]])

        self.init_state = tf.placeholder(tf.float32,[None,self.num_units],name="initial_state")

        self.fused_cell = LDS_Cell(self.num_units,cell_clip=clip_value)

        cell_out,self.final_state = tf.nn.dynamic_rnn(self.fused_cell,head,initial_state = self.init_state,time_major=True)

        cell_out = tf.reshape(cell_out,[-1,self.num_units])

        y = tf.keras.layers.Dense(self.action_size,activation=None,name="ct_out")(cell_out)
        # Reshape back to sequenced batch form
        self.y = tf.reshape(y,shape=[tf.shape(self.x_obs)[0],tf.shape(self.x_obs)[1],self.action_size])

        #Output
        tf.identity(self.y,name='prediction')
        tf.identity(self.final_state,name='final_state')

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(self.target_y, self.y)),axis=-1))

        self.add_stablity_optimization()

        # Loss, error and training algorithm
        self.mean_abs_error = tf.reduce_mean(tf.abs(tf.subtract(self.y, self.target_y)))

        optimizer = tf.train.AdamOptimizer(0.0001)
        self.train_step = optimizer.minimize(self.loss)

    def zero_state(self,batch_size):
        return np.zeros([batch_size,self.num_units],dtype=np.float32)

    def share_sess(self, sess):
        self.sess = sess


    def is_stable(self):
        A = self.sess.run(self.fused_cell.W)
        e,v = np.linalg.eig(A)
        stable = True
        for i in range(e.shape[0]):
            r = np.real(e[i])
            if(r > 0.0):
                stable = False
        return stable

    def add_stablity_optimization(self):
        diag_part = tf.diag_part(self.fused_cell.W)
        row_reduced = tf.reduce_sum(tf.abs(self.fused_cell.W),axis=1)
        abs_diag = tf.abs(diag_part)

        # Summation trick to avoid summing over the diagonal
        R = row_reduced-abs_diag

        # Upper bound on lambda on real axis
        lambda_ub = diag_part + R + 0.00000001

        self.surrogate_loss = tf.reduce_sum(tf.nn.relu(lambda_ub))
        optimizer = tf.train.AdamOptimizer(0.0001)
        self.opt_stability = optimizer.minimize(self.surrogate_loss)



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

        w_path = os.path.join(path,"w.csv")
        A = self.sess.run(self.fused_cell.W)

        np.savetxt(w_path,A)

        e,v = np.linalg.eig(A)
        w_info = os.path.join(path,"eigen.txt")
        with open(w_info,"w") as f:
            f.write("Eigenvalues: "+str(e))
            

    def restore_from_checkpoint(self, path):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(path,'-model'))

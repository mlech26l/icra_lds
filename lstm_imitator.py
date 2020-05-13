import tensorflow as tf
import os
import numpy as np

class LSTM_Imitator:

    def __init__(self,lstm_size,clip_value,lidar_size,state_size):

        self.lidar_size = lidar_size
        self.state_size = state_size

        self.lstm_size = lstm_size

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

        self.init_c = tf.placeholder(tf.float32,[None,self.lstm_size],name="initial_state_c")
        self.init_h = tf.placeholder(tf.float32,[None,self.lstm_size],name="initial_state_h")

        self.init_tuple =  tf.nn.rnn_cell.LSTMStateTuple(self.init_c,self.init_h)

        cell_clip = clip_value if clip_value > 0 else None
        self.fused_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size,cell_clip=cell_clip)

        lstm_out,self.final_state = tf.nn.dynamic_rnn(self.fused_cell,head,initial_state = self.init_tuple,time_major=True)

        lstm_out = tf.reshape(lstm_out,[-1,self.lstm_size])
        
        # flatten LSTM output for dense layer to merge lstm output to the inverse_r output
        y = tf.keras.layers.Dense(1,activation=None)(lstm_out)
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
        return tf.contrib.rnn.LSTMStateTuple(
            np.zeros([batch_size,self.lstm_size],dtype=np.float32),
            np.zeros([batch_size,self.lstm_size],dtype=np.float32),
        )

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
            self.init_tuple: init_state}

        prediction,next_state = self.sess.run([self.y,self.final_state], feed_dict=feed_dict)
        return float(prediction.flatten()),next_state

    
    def evaluate(self, batch_x_state, batch_x_lidar,batch_y):
        feed_dict = {
            self.x_state: batch_x_state,
            self.x_lidar: batch_x_lidar,
            self.target_y: batch_y,
            self.init_tuple:self.zero_state(batch_x_state.shape[1])
            }

        loss,mae = self.sess.run([self.loss,self.mean_abs_error], feed_dict=feed_dict)

        return loss,mae
    def train_iter(self, batch_x_state, batch_x_lidar,batch_y):
        feed_dict = {
            self.x_state: batch_x_state,
            self.x_lidar: batch_x_lidar,
            self.target_y: batch_y,
            self.init_tuple:self.zero_state(batch_x_state.shape[1])
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



class LSTM_Cheetah:

    def __init__(self,lstm_size,clip_value,obs_size,action_size):

        self.obs_size = obs_size
        self.action_size = action_size

        self.lstm_size = lstm_size

        self.x_obs = tf.placeholder(tf.float32, shape=[None, None, self.obs_size],name='state')
        self.target_y = tf.placeholder(tf.float32, shape=[None,None,self.action_size])

        head = self.x_obs
        head = tf.reshape(self.x_obs,[-1,self.obs_size])
        head = tf.keras.layers.Dense(128,activation="relu")(head)
        head = tf.keras.layers.Dense(128,activation=None)(head)
        head = tf.reshape(head,shape=[tf.shape(self.x_obs)[0],tf.shape(self.x_obs)[1],head.shape[-1]])

        self.init_c = tf.placeholder(tf.float32,[None,self.lstm_size],name="initial_state_c")
        self.init_h = tf.placeholder(tf.float32,[None,self.lstm_size],name="initial_state_h")

        self.init_tuple =  tf.nn.rnn_cell.LSTMStateTuple(self.init_c,self.init_h)

        cell_clip = clip_value if clip_value > 0 else None
        self.fused_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size,cell_clip=cell_clip)

        lstm_out,self.final_state = tf.nn.dynamic_rnn(self.fused_cell,head,initial_state = self.init_tuple,time_major=True)

        lstm_out = tf.reshape(lstm_out,[-1,self.lstm_size])

        # flatten LSTM output for dense layer to merge lstm output to the inverse_r output
        y = tf.keras.layers.Dense(self.action_size,activation=None,name="ct_out")(lstm_out)
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
        return tf.contrib.rnn.LSTMStateTuple(
            np.zeros([batch_size,self.lstm_size],dtype=np.float32),
            np.zeros([batch_size,self.lstm_size],dtype=np.float32),
        )

    def share_sess(self, sess):
        self.sess = sess


    def predict_step(self, x_obs,init_state=None):
        if(init_state is None):
            init_state = self.zero_state(1)

        # Reshape sequence into a batch of 1 sequence
        x_obs = x_obs.reshape([1,1,self.obs_size])
        feed_dict = {
            self.x_obs: x_obs,
            self.init_tuple: init_state}

        prediction,next_state = self.sess.run([self.y,self.final_state], feed_dict=feed_dict)
        return prediction.flatten(),next_state

    
    def evaluate(self, batch_x_obs, batch_action):
        feed_dict = {
            self.x_obs: batch_x_obs,
            self.target_y: batch_action,
            self.init_tuple:self.zero_state(batch_x_obs.shape[1])
            }

        loss,mae = self.sess.run([self.loss,self.mean_abs_error], feed_dict=feed_dict)

        return loss,mae
    def train_iter(self,  batch_x_obs, batch_action):
        feed_dict = {
            self.x_obs: batch_x_obs,
            self.target_y: batch_action,
            self.init_tuple:self.zero_state(batch_x_obs.shape[1])
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

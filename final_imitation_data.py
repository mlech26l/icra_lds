import numpy as np
import os

class FinalImitationData:

    def __init__(self,seq_len=32):
        all_files = sorted([os.path.join("training_data",d) for d in os.listdir("training_data") if d.endswith(".csv")])

        np.random.RandomState(1239234).shuffle(all_files)
        valid_files = all_files[:5]
        train_files = all_files[5:]

        self.train_x, self.train_y = self._load_files(train_files)
        self.valid_x, self.valid_y = self._load_files(valid_files)

        self.test_x, self.test_y = self.valid_x,self.valid_y

        self.state_size = self.train_x[0][0].shape[1]
        self.lidar_size = self.train_x[0][1].shape[1]

        self.seq_len = seq_len
        self.batch_size = 32

    def _sample_set(self,batch_size,set_x,set_y,rng=np.random,seq_len=None):
        if(seq_len is None):
            seq_len = self.seq_len
        x_state = np.empty([seq_len,batch_size,self.state_size])
        x_lidar = np.empty([seq_len,batch_size,self.lidar_size])
        y = np.empty([seq_len,batch_size,1])
        
        for b in range(batch_size):
            b_i = rng.randint(len(set_y))

            t_start = rng.randint(set_y[b_i].shape[0]-seq_len)

            x_state[:,b] = set_x[b_i][0][t_start:t_start+seq_len]
            x_lidar[:,b] = set_x[b_i][1][t_start:t_start+seq_len]
            y[:,b,0] = set_y[b_i][t_start:t_start+seq_len]

        return (x_state,x_lidar,y)

    def sample_training_set(self):
        return self._sample_set(self.batch_size,self.train_x,self.train_y)

    def sample_validation_set(self):
        return self._sample_set(256,self.valid_x,self.valid_y,rng=np.random.RandomState(12309),seq_len=32)

    def _augment_data(self,x_lidar,x_state,y):
        x2_lidar = x_lidar[:,::-1]
        x2_state = -x_state

        y2 = -y

        return (x2_lidar,x2_state,y2)

    def _load_files(self,files):
        all_x = []
        all_y = []
        for f in files:
           
            arr = np.loadtxt(f,delimiter=',')
            x_state = arr[:,1:3].astype(np.float32)
            x_lidar = arr[:,3:-1].astype(np.float32)
            y = arr[:,-1].astype(np.float32)

            all_x.append((x_state,x_lidar))
            all_y.append(y)

            x2_lidar,x2_state,y2 = self._augment_data(x_lidar,x_state,y)
            all_x.append((x2_state,x2_lidar))
            all_y.append(y2)
            print("Loaded file '{}' of length {:d}".format(f,x_state.shape[0]))
        return all_x,all_y

    def iterate_test(self):
        for i in range(len(self.test_y)):
            for t in range(self.test_y[i].shape[0]):
                x_state = self.test_x[i][0][t]
                x_lidar = self.test_x[i][1][t]
                y = self.test_y[i][t]
                reset = False
                if(t == 0):
                    reset = True
                yield(x_state,x_lidar,y,reset)

if __name__ == "__main__":
    data = FinalImitationData()
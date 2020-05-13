import numpy as np
import os

class CheetahData:

    def __init__(self,seq_len):
        all_files = sorted([os.path.join("cheetah",d) for d in os.listdir("cheetah") if d.endswith(".csv")])

        train_files = all_files[5:25]
        valid_files = all_files[:5]

        self.seq_len = seq_len
        self.obs_size = 17
        self.action_size = 6
        self.batch_size = 32

        self._load_files(all_files)
        self.train_x, self.train_y = self._load_files(train_files)
        self.valid_x, self.valid_y = self._load_files(valid_files)

        
        self.batch_size=1024
        all_x = self.sample_training_set()[0].reshape([-1,self.obs_size])
        mean_x = np.mean(all_x,axis=0)
        std_x = np.std(all_x,axis=0)
        print("mean_x: ",str(mean_x))
        print("std_x: ",str(std_x))

    def _sample_set(self,batch_size,set_x,set_y,rng=np.random,seq_len=None):
        if(seq_len is None):
            seq_len = self.seq_len
        obs = np.empty([seq_len,batch_size,self.obs_size])
        actions = np.empty([seq_len,batch_size,self.action_size])
        
        for b in range(batch_size):
            b_i = rng.randint(len(set_y))

            t_start = rng.randint(set_y[b_i].shape[0]-seq_len)

            obs[:,b] = set_x[b_i][t_start:t_start+seq_len]
            actions[:,b] = set_y[b_i][t_start:t_start+seq_len]

        return (obs,actions)

    def sample_training_set(self):
        return self._sample_set(self.batch_size,self.train_x,self.train_y)

    def sample_validation_set(self):
        return self._sample_set(256,self.valid_x,self.valid_y,rng=np.random.RandomState(12309))

    def _load_files(self,files):
        all_x = []
        all_y = []
        all_rewards = []
        for f in files:
           
            arr = np.loadtxt(f,delimiter=',')
            obs = arr[:,:self.obs_size].astype(np.float32)
            actions = arr[:,self.obs_size:-1].astype(np.float32)
            r = arr[-1,-1].astype(np.float32)

            all_x.append(obs)
            all_y.append(actions)
            all_rewards.append(r)

            print("Loaded file '{}' of length {:d}".format(f,obs.shape[0]))
        print("Loaded {:d} files with mean return {:0.2f} +- {:0.2f}".format(len(all_rewards),np.mean(all_rewards),np.std(all_rewards)))
        return all_x,all_y


if __name__ == "__main__":
    data = CheetahData(seq_len=64)
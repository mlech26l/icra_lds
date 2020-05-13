import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # train on CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Uncomment to hide tensorflow logs
import tensorflow as tf
import numpy as np
import argparse
from cheetah_data import CheetahData
from lstm_imitator import LSTM_Cheetah
from ctrnn_imitator import CTRNN_Cheetah
from lds_imitator import LDS_Cheetah
import gym

# Parse arugments
parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--model', default='lstm')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seq_len', type=int, default=32)
args = parser.parse_args()


data = CheetahData(seq_len=args.seq_len)
valid_data = data.sample_validation_set()

if(args.model == "lstm"):
    model = LSTM_Cheetah(lstm_size=32,clip_value=10,obs_size=data.obs_size,action_size=data.action_size)
elif(args.model == "lds" or args.model == "linear"):
    model = LDS_Cheetah(num_units=32,clip_value=-1,obs_size=data.obs_size,action_size=data.action_size)
elif(args.model == "ctrnn"):
    model = CTRNN_Cheetah(num_units=32,clip_value=10,obs_size=data.obs_size,action_size=data.action_size)
else:
    raise ValueError("Unknown model")


sess = tf.Session()
sess.run(tf.global_variables_initializer())
model.share_sess(sess)

tf_vars = tf.trainable_variables()
print("#### Trainable variables #####")
for v in tf_vars:
    print("   - {}".format(str(v.name)))
print("##############################")
    
if(args.model == "lds"):
    opt_round = 0
    while not model.is_stable():
        print("model not stable optimize ({:d}) ... ".format(opt_round))
        opt_round += 1
        sess.run(model.opt_stability)
        
i = 0
while True:
    base_path = os.path.join("sessions_cheetah","{}_{:04d}".format(args.model,i))
    if(not os.path.exists(base_path)):
        break
    i += 1
    
if(not os.path.exists(base_path)):
    os.makedirs(base_path)
    

with open(os.path.join(base_path,"training_log.csv"),"w") as f:
    f.write("epoch,train_loss, train_mae, valid_loss, valid_mae\n")

best_epoch = 0
best_valid_loss = np.PINF
for epoch in range(args.epochs):
    valid_loss,valid_mae = model.evaluate(valid_data[0],valid_data[1])

    if(valid_loss < best_valid_loss):
        best_epoch = epoch
        best_valid_loss = valid_loss
        model.create_checkpoint(base_path)

    train_loss, train_mae = [], []
    for i in range(50):
        x_obs,action = data.sample_training_set()
        loss,mae = model.train_iter(x_obs,action)
        train_loss.append(loss)
        train_mae.append(mae)
    
    if(args.model == "lds"):
        opt_round = 0
        while not model.is_stable():
            print("model not stable optimize ({:d}) ... ".format(opt_round))
            opt_round += 1
            sess.run(model.opt_stability)

    with open(os.path.join(base_path,"training_log.csv"),"a") as f:
        f.write("{}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
            epoch,
            np.mean(train_loss),np.mean(train_mae),
            valid_loss,valid_mae
        ))

    print("Epochs {:03d}/{:03d}, train loss: {:0.3f}, mae: {:0.3f}, valid loss: {:0.3f}, mae: {:0.3f}".format(
        epoch,args.epochs,
        np.mean(train_loss),np.mean(train_mae),
        valid_loss,valid_mae
    ))

print("Best epoch: {:03d} with valid loss: {:0.3f}".format(best_epoch,best_valid_loss))
with open(os.path.join(base_path,"best_epoch.txt"),"w") as f:
    f.write("best epopch: {:d}, valid loss: {:0.5f}".format(
        best_epoch,best_valid_loss
    ))

model.restore_from_checkpoint(base_path)

N = 10
env =  gym.make("HalfCheetah-v2")

total_rewards = []
for i in range(N):
    r_sum = 0
    obs = env.reset()
    rnn_state = None
    done = False

    while not done:
        action,rnn_state = model.predict_step(obs,rnn_state)
        action[np.isnan(action)] = 0
        action = np.clip(action,env.action_space.low,env.action_space.high)
        obs, reward, done, info = env.step(action)

        r_sum += reward

    total_rewards.append(r_sum)


np.savetxt(os.path.join(base_path,"results.txt"),np.array([best_epoch,best_valid_loss,np.mean(total_rewards)],dtype=np.float32))

print("Mean rollout: {:0.2f} +- {:0.2f}".format(np.mean(total_rewards),np.std(total_rewards)))
with open(os.path.join(base_path,"rollouts.txt"),"w") as f:
    f.write("Mean: {:0.2f} +- {:0.2f}\ndetails:\n".format(np.mean(total_rewards),np.std(total_rewards)))
    for r in total_rewards:
        f.write("{:0.2f}\n".format(r))

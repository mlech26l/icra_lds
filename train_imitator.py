import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # train on CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Uncomment to hide tensorflow logs
import tensorflow as tf
import numpy as np
import argparse
from imitation_data import ImitationData
from final_imitation_data import FinalImitationData
from lstm_imitator import LSTM_Imitator
from ctrnn_imitator import CTRNN_Imitator
from lds_imitator import LDS_Imitator

# Parse arugments
parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--model', default='lstm')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seq_len', type=int, default=32)
parser.add_argument('--real', action="store_true")
args = parser.parse_args()


if(args.real):
    data = FinalImitationData(seq_len = args.seq_len)
else:
    data = ImitationData(seq_len=args.seq_len)
valid_data = data.sample_validation_set()

if(args.model == "lstm"):
    model = LSTM_Imitator(lstm_size=32,clip_value=10,lidar_size=data.lidar_size,state_size=data.state_size)
elif(args.model == "lds" or args.model == "linear"):
    model = LDS_Imitator(num_units=32,clip_value=-1,lidar_size=data.lidar_size,state_size=data.state_size)
elif(args.model == "ctrnn"):
    model = CTRNN_Imitator(num_units=32,clip_value=10,lidar_size=data.lidar_size,state_size=data.state_size)
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
    if(args.real):
        base_path = os.path.join("real_sessions","{}_{:04d}".format(args.model,i))
    else:
        base_path = os.path.join("sessions","{}_{:04d}".format(args.model,i))
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
    valid_loss,valid_mae = model.evaluate(valid_data[0],valid_data[1],valid_data[2])

    if(valid_loss < best_valid_loss):
        best_epoch = epoch
        best_valid_loss = valid_loss
        model.create_checkpoint(base_path)

    train_loss, train_mae = [], []
    for i in range(5):
        x_state,x_lidar,y = data.sample_training_set()
        loss,mae = model.train_iter(x_state,x_lidar,y)
        train_loss.append(loss)
        train_mae.append(mae)
    
    if(args.model == "lds"):
        opt_round = 0
        while not model.is_stable():
            print("model not stable optimize ({:d}) ... ".format(opt_round))
            opt_round += 1
            sess.run(model.opt_stability)

    with open(os.path.join(base_path,"training_log.csv"),"a") as f:
        f.write("{}, {:0.5f}, {:0.5f}, {:0.5f}, {:0.5f}\n".format(
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

test_errors = []
rnn_state = None
for x_state,x_lidar,y,reset in data.iterate_test():
    if(reset):
        rnn_state = None
    prediction,rnn_state = model.predict_step(x_state,x_lidar,rnn_state)
    test_errors.append(prediction-y)

test_loss = np.mean(np.square(test_errors))
test_mae = np.mean(np.abs(test_errors))
np.savetxt(os.path.join(base_path,"final_metrics.csv"),np.array([best_epoch,test_loss,test_mae]))
import os
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# plot parameters 
plt.rcParams["figure.figsize"] = (60,30)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = True

# file list 읽어오기
# 2번 전극 조합 1번 marker 위치 
path_dir = 'D:/CSJ/Tensorflow/market_price_(EMG)_lstm/Data/example/norm' #파일 저장된 위치
file_list = os.listdir(path_dir)


# file list 만들기
all_file_list = []
for x in range(len(file_list)):
    all_file_list.append(path_dir + "/" + file_list[x])
# file 순서 random shuffle    
random.shuffle(all_file_list)
        
# train/test data set 나누기
ratio = 0.7
num = round(len(all_file_list) * ratio)
train_file_list, test_file_list = all_file_list[:num], all_file_list[num:]


# train data load
train_data = [[0,0,0,0,0,0,0,0,0,0]]
for i in range(len(train_file_list)):
    temp= np.loadtxt(train_file_list[i],delimiter=",",dtype=np.float32)
    train_data = np.append(train_data, temp, axis=0)    
train_data = train_data[1:,:]

#test data load
test_data = [[0,0,0,0,0,0,0,0,0,0]]
for i in range(len(test_file_list)):
    temp = np.loadtxt(test_file_list[i],delimiter=",",dtype=np.float32)
    test_data = np.append(test_data, temp, axis=0)    
test_data = test_data[1:,:]

# Hyper Parameters
input_data_column_cnt = 4  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 1 # 결과데이터의 컬럼 개수 
seq_length = 30            # 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 10    # 각 셀의 (hidden)출력 크기
forget_bias = 1.0          # 망각편향(기본값 1.0)
num_stacked_layers = 1     # stacked LSTM layers 개수
keep_prob = 1.0            # dropout할 때 keep할 비율 
epoch_num = 300      # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
learning_rate = 0.01       # 학습률
y_value = 4                # y의 x y z az el r


# build a train dataset
trainX = []
trainY = []
for i in range(0, len(train_data) - seq_length):
    _x = train_data[i:i + seq_length,0:4]
    _y = train_data[i + seq_length,y_value]  
    print(_x, "->", _y)
    trainX.append(_x)
    trainY.append(_y)
trainX = np.array(trainX)
trainY = np.array([np.array(trainY)]).T

# build a validation train dataset
v_train_data = train_data[0:round(len(train_data)*0.3),:]
v_trainX = []
v_trainY = []
for i in range(0, len(v_train_data) - seq_length):
    _x = v_train_data[i:i + seq_length,0:4]
    _y = v_train_data[i + seq_length,y_value]  
    print(_x, "->", _y)
    v_trainX.append(_x)
    v_trainY.append(_y)
v_trainX = np.array(v_trainX)
v_trainY = np.array([np.array(v_trainY)]).T
    
# build a test dataset
testX = []
testY = []
for i in range(0, len(test_data) - seq_length):
    _x = test_data[i:i + seq_length,0:4]
    _y = test_data[i + seq_length,y_value]  
    print(_x, "->", _y)
    testX.append(_x)
    testY.append(_y)
testX = np.array(testX)
testY = np.array([np.array(testY)]).T


# input X, output Y를 생성한다
X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
Y = tf.placeholder(tf.float32, [None, 1])

# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, 1]) 
predictions = tf.placeholder(tf.float32, [None, 1])

# 모델(LSTM 네트워크) 생성
def lstm_cell():
    # LSTM셀을 생성
    # num_units: 각 Cell 출력 크기
    # forget_bias:  to the biases of the forget gate 
    #              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim, 
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell

# num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

# LSTM Cell들을 연결
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)
 
#LSTM RNN의 마지막 (hidden)출력만을 사용했다.
hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)   

# 손실함수로 평균제곱오차를 사용한다
loss = tf.reduce_sum(tf.square(hypothesis - Y))

#tensorboard용 loss 저장
cost_summ = tf.summary.scalar("cost",loss)
merged = tf.summary.merge_all()

    
# 최적화함수로 AdamOptimizer를 사용한다
optimizer = tf.train.AdamOptimizer(learning_rate)               
train = optimizer.minimize(loss)

# RMSE(Root Mean Square Error)
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

train_error_summary = [] # 학습용 데이터의 오류를 중간 중간 기록한다
test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다
test_predict = ''        # 테스트용데이터로 예측한 결과
validation_train_predict = '' # validation 예측

 
sess = tf.Session()

# graph 관련 정보를 log폴더에 저장
writer = tf.summary.FileWriter("D:/CSJ/Tensorflow/Lstm regression/logs",sess.graph)
#tensorboard --logdir=D:/CSJ/Tensorflow/Lstm regression/logs


sess.run(tf.global_variables_initializer())
 
# 학습한다

print('학습을 시작합니다...')
for epoch in range(epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if ((epoch+1) % 100 == 0) or (epoch == epoch_num-1): # 100번째마다 또는 마지막 epoch인 경우
        # 학습용데이터로 rmse오차를 구한다
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})
        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
        train_error_summary.append(train_error)
 
        # 테스트용데이터로 rmse오차를 구한다
        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        test_error_summary.append(test_error)
        
        # Validation 데이터로 예측 잘되는지 확인
        validation_train_predict = sess.run(hypothesis, feed_dict={X: v_trainX})
        
        #
        summary = sess.run(merged, feed_dict={X: trainX, Y: trainY})
        writer.add_summary(summary,epoch)
        
        # 현재 오류를 출력한다
        print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch+1, train_error, test_error, test_error-train_error))
    

 # 결과 그래프 출력
plt.figure(1)
plt.plot(train_error_summary)
plt.plot(test_error_summary)
plt.xlabel('Epoch(x100)')
plt.ylabel('Root Mean Square Error')       
        
plt.figure(2)
plt.plot(v_trainY)
plt.plot(validation_train_predict)
plt.xlabel("Time")
plt.ylabel("value")
        
plt.figure(3)
plt.plot(testY)
plt.plot(test_predict)
plt.xlabel("Time")
plt.ylabel("value")
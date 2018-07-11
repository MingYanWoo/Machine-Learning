# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

train_data = pd.read_csv('train.csv')
train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data = train_data.fillna(0)

train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 0
train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1
train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0
train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1
train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2

# 丢弃Cabin标签
train_data.drop(['Cabin'], axis=1, inplace=True)
train_data['Deceased'] = train_data['Survived'].apply(lambda s: 1 - s)

dataset_X = train_data[['Sex','Age','Pclass','SibSp','Parch','Fare','Embarked']]
dataset_Y = train_data[['Survived']]

# 划分训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(dataset_X.as_matrix(),
                                                  dataset_Y.as_matrix(),
                                                  test_size=0,
                                                  random_state=42)

# 归一化
X_train= preprocessing.MinMaxScaler().fit_transform(X_train)

print('X_train: ', X_train.shape)
print('Y_train: ', Y_train.shape)

X = tf.placeholder(tf.float32, shape=[None, 7])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# layer1
W1 = tf.Variable(tf.random_normal([7, 7]), name='weight1')
b1 = tf.Variable(tf.random_normal([7]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

# layer2
W2 = tf.Variable(tf.random_normal([7, 7]), name='weight2')
b2 = tf.Variable(tf.random_normal([7]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# layer3
W3 = tf.Variable(tf.random_normal([7, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')

learning_rate = 0.1

# 假设函数
hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)
# cost
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    feed = {X: X_train, Y: Y_train}
    for step in range(10001):
        sess.run(train, feed_dict=feed)
        if step % 1000 == 0:
            saver.save(sess, 'model/titanic', global_step=step)
            print(step, sess.run(cost, feed_dict=feed))

    # Accuracy report
    a = sess.run(accuracy, feed_dict=feed)
    print "Learning rate: %f, Accuracy: %f" % (learning_rate, a)




    # 测试
    test_data = pd.read_csv('test.csv')
    test_data['Embarked'] = test_data['Embarked'].fillna('S')
    test_data = test_data.fillna(0)

    test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0
    test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1
    test_data.loc[test_data['Embarked'] == 'S', 'Embarked'] = 0
    test_data.loc[test_data['Embarked'] == 'C', 'Embarked'] = 1
    test_data.loc[test_data['Embarked'] == 'Q', 'Embarked'] = 2

    # 丢弃Cabin标签
    test_data.drop(['Cabin'], axis=1, inplace=True)

    testset_X = test_data[['Sex','Age','Pclass','SibSp','Parch','Fare','Embarked']]

    # 归一化
    testset_X= preprocessing.MinMaxScaler().fit_transform(testset_X)

    test_predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    p = sess.run(test_predicted, feed_dict={X: testset_X})
    # 二维转换成一维
    p = np.reshape(p, len(p))
    # tf.float32 转 tf.int16
    p = tf.cast(p, dtype=tf.int16)
    # tensor 转 数组
    p = p.eval()
    # 保存结果
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": p
    })

    submission.to_csv("titanic-NN-submission.csv", index=False)


    # 正确率
    official_data = pd.read_csv('gender_submission.csv')
    my_data = pd.read_csv('titanic-NN-submission.csv')
    offArr = official_data['Survived']
    myArr = my_data['Survived']
    counter = 0.0
    for i in range(len(offArr)):
        # print(offArr[i], myArr[i])
        if offArr[i] == myArr[i]:
            counter = counter + 1

    print 'test_accuracy: ', float(counter/len(offArr))
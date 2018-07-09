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
                                                  test_size=0.2,
                                                  random_state=42)

# 归一化
X_train= preprocessing.MinMaxScaler().fit_transform(X_train)

print('X_train: ', X_train.shape)
print('Y_train: ', Y_train.shape)

W = tf.Variable(tf.random_normal([7, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
X = tf.placeholder(tf.float32, shape=[None, 7])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 假设函数
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# cost
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    feed = {X: X_train, Y: Y_train}
    for step in range(1001):
        sess.run(train, feed_dict=feed)
        if step % 100 == 0:
            saver.save(sess, 'model/titanic', global_step=step)
            print(step, sess.run(cost, feed_dict=feed))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict=feed)
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)




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

    print('testset_X: ', testset_X.shape)
    test_predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    test_accuracy = tf.reduce_mean(tf.cast(tf.equal(test_predicted, Y), dtype=tf.float32))
    h, p = sess.run([hypothesis, test_predicted], feed_dict={X: testset_X})
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

    submission.to_csv("titanic-submission.csv", index=False)
from pyparsing import delimitedList
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/train.csv")

df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df = df.dropna()
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
df['Port'] = df['Embarked'].map({'C': 0, 'S': 1, 'Q': 2}).astype(int)
df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]

#read test data
test_data = pd.read_csv("data/test.csv")

test_data = test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_data = test_data.dropna()
test_data['Gender'] = test_data['Sex'].map({'female': 0, 'male': 1}).astype(int)
test_data['Port'] = test_data['Embarked'].map({'C': 0, 'S': 1, 'Q': 2}).astype(int)
test_data = test_data.drop(['Sex', 'Embarked'], axis=1)
test_data_columns = test_data.columns
test_data = test_data.as_matrix()


# parameters
learning_rate = 0.005
training_epochs = 2000
batch_size = 100
display_step = 1

# Network parameters
n_hidden_1 = 32
n_hidden_2 = 32
n_input = df.shape[1] - 2
n_classes = 2
real_input = df.as_matrix()
tout = real_input[:, 0:1]
tout = np.stack((1 - tout[:, 0], tout[:, 0]), axis=-1)
# print(tout)
# print("HELL!")
# print(real_input[:,0:1])
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


def multilayer_perceptron(x, weights, biases):
    # HIDDEN layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.softmax(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.softmax(layer_2)
    # output layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='l1_weights'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='l2_weights'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='out_weights')
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='l1_biases'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='l2_biases'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='out_biases')
}

# model
pred = multilayer_perceptron(x, weights, biases)

# Define loss optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# cost = tf.reduce_sum(tf.square(tf.subtract(pred, y)), [0, 1])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# init all variables
init = tf.global_variables_initializer()
# print(n_input)
# print(cols)
# print(df['Embarked'].unique())
# print(df.head())
# print(df.info())
# print(df.shape[1])
with tf.Session() as session:
    # session = tf_debug.LocalCLIDebugWrapperSession(session)
    session.run(init)
    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("/tmp/titanic", session.graph)
    init = tf.global_variables_initializer()
    for epoch in range(training_epochs):
        _, c = session.run([optimizer, cost], feed_dict={x: real_input[:, 2:], y: tout})
        # saver = tf.train.Saver()
        # save_path = saver.save(session, "/tmp/model.ckpt")
        # print("Model saved in path : %s " % save_path)
        # print(c)

    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    output_value = tf.nn.softmax(pred)
    predictions = output_value.eval({x:test_data[:,1:]})
    predicted_data = tf.concat([tf.cast(test_data, "float"), output_value], 1)
    print("Training Accuracy:", accuracy.eval({x:real_input[:,2:], y: tout}))
    test_predictions = predicted_data.eval({x:test_data[:,1:]})
    # test_predictions = np.concatenate(test_data_columns, test_predictions)
    print(test_predictions[:5, :])
    np.savetxt("predictions.csv", test_data_columns, delimiter=",", fmt='%s')
    f = open('predictions.csv', 'ab')
    np.savetxt(f,test_predictions, delimiter=",", fmt='%f')

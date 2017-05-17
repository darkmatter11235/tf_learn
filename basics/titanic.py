import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/train.csv")
df = df.dropna()
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)
df['Port'] = df['Embarked'].map({'C':0, 'S':1, 'Q':2}).astype(int)
df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]

#parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

#Network parameters
n_hidden_1 = 256
n_hideen_2 = 256
n_input = df.shape[1]-1
n_classes = 2
real_input = df.as_matrix()
print(real_input.shape)
#tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def multilayer_perceptron(x, weights, biases):
	#HIDDEN layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	#output layer
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer

weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hideen_2])),
		'out': tf.Variable(tf.random_normal([n_hideen_2, n_classes]))
}

biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hideen_2])),
		'out': tf.Variable(tf.random_normal([n_classes]))
}

#model
pred = multilayer_perceptron(x, weights, biases)

#Define loss optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#init all variables
init = tf.global_variables_initializer()
#print(n_input)
#print(cols)
#print(df['Embarked'].unique())
#print(df.head())
#print(df.info())
#print(df.shape[1])
with tf.Session() as session:
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter("/tmp/titanic", session.graph)
	init = tf.global_variables_initializer()
	_, c = session.run([optimizer, cost], feed_dict={x: real_input[2:], y: real_input[1]})
	print(c)


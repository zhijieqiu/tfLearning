import numpy as np
import tensorflow as tf


W = tf.Variable(initial_value=[0.3],dtype=tf.float32)
b = tf.Variable(initial_value=[-0.3],dtype=tf.float32)

x = tf.placeholder(dtype=tf.float32)
linear_model = W*x+b

y = tf.placeholder(dtype=tf.float32)
loss = tf.reduce_sum(tf.square(linear_model-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss)
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train,feed_dict={x:x_train,y:y_train})

curr_W,curr_b,curr_x = sess.run([W,b,loss],feed_dict={x:x_train,y:y_train})
print("W:%s b:%s x:%s"%(curr_W,curr_b,curr_x))
a = input()
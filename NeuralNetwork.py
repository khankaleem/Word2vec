import ReadFile as RF
import tensorflow as tf
import numpy as np

x_train, y_train, vocab_size, int_to_word, word_to_int = RF.GetData()

#define the architecture for the neural network
EMBEDDING_DIM = 5
eta = 0.1
epochs = 1000

#Make placeholders for x and y
x = tf.placeholder(tf.float32, shape = (None, vocab_size))
y_label = tf.placeholder(tf.float32, shape = (None, vocab_size))
#define the embedding layer
w1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))
#define output for embedding layer
hidden_representation = tf.add(tf.matmul(x, w1), b1)
#define the ouput layer
w2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
#define output for final layer
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, w2), b2))
#define the loss function
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label*tf.log(prediction), reduction_indices = [1]))
#define the train step
train_step = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy_loss)
#run session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#train for epochs
for epoch in range(epochs):
    sess.run(train_step, feed_dict = {x: x_train, y_label: y_train})
    print('epoch: ' + str(epoch+1))
    print('loss: ', sess.run(cross_entropy_loss, feed_dict = {x: x_train, y_label: y_train}))
    print()
    
#Get Embedding    
EmbeddedVectors = sess.run(w1+b1)

#function for word clustering
def EuclideanDistance(vec1, vec2):
    return np.sqrt(np.sum(vec1 - vec2)**2)

def ClosestWords(word_index, word_embedding):
    radius = 100
    closest_words = []
    
    query_vector = word_embedding[word_index]
    for index, embedding in enumerate(word_embedding):
        if EuclideanDistance(query_vector, embedding) < radius and not np.array_equal(query_vector, embedding):
            closest_words.append((int_to_word[index], EuclideanDistance(query_vector, embedding)))
    
    return closest_words
    
print(ClosestWords(word_to_int['kaleem'], EmbeddedVectors))


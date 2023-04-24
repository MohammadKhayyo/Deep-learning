Student Name: Mohammad Khayyo


== Program Description ==
The program will solve a classification problem with the data set.  The data set has the following characteristics: the data is divided into two groups, the "positive" group or the "negative" group.  Each line in the file indicates a blood test of one person for testing. There are 9 characteristics, the values ​​in each of the characteristics can be: one, two, and three.  And a blood test can come out positive or negative.  These are the results of blood tests, when some people received a positive answer and for the other part the answer came out negative.  - A blood test can be positive, indicating the information on findings of a certain disease or negative, which is not such.


== The Best Prediction Info ==
Weights: [-0.25446298, 0.04715692, 0.04716542, 0.0471482, 0.0471775, 0.04714226, 0.04716339, 0.04715219, 0.04715646, 0.04714365]
accuracy : 76.30480167014613%


== functions ==
1) def preprocess_data(filename, x_0=1):
This function reads the file and converts its data into a matrix,then it shuffles these lines ,and then separates the training and labels. then return The training and labels.
2) def train(data_for_train, expected_y, learning_rate=0.0001, threshold=0.5, iterations_num=1500):
the function trains the model according to randomly chosen weights value between (-1, 1) and the data.
3)def predict(data_for_train, expected_y, weights, threshold=0.0):
the function predict the result of the given inputs and calculate the accuracy.

== Program Files ==
1. perceptron.py


== How To Run In Terminal? ==

run: python perceptron.py
optional way to run : python perceptron.py filename
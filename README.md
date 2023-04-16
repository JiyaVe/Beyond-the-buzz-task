# Beyond-the-buzz-task
The following code performs a classification task using a Multi-Layer Perceptron (MLP) classifier from the Scikit-Learn library. I used a neural network for creation of classifier.

The first step is to import the necessary libraries and modules: MLPClassifier from sklearn.neural_network, make_classification from sklearn.datasets, pandas for raeding csv files, SMOTE from imblearn.over_sampling for preprocessing, and accuracy_score from sklearn.metrics to calculate accuracy.

Then, the training and test data are read from the CSV files using pandas, and the features and target values are separated into train_x, train_y to train the model , test_x, and test_y variables to later test the model.

After that, the SMOTE  is applied to balance the sample dataset.

Next, an instance of MLPClassifier is created with two hidden layers, each with 100 neurons, and the maximum number of iterations is set to 100 since it gave a good accuracy after trying other numbers. The fit method is called on the MLPClassifier object to train the model on the balanced dataset.

The predict method is used to predict the target values for the test dataset, and the accuracy of the predictions is evaluated using the accuracy_score function from the sklearn.metrics module. The results are printed.

Finally, the predicted target values are saved to a CSV file using pandas into predictions.csv.
The math used here is default one ,RELU as it gave a good accuracy.

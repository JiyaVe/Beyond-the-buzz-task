from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
# >>> from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sampleSubmission = pd.read_csv('sampleSubmission.csv')
# print (train_data.head())

train_x = train_data.drop('VERDICT', axis=1).values
train_y = train_data['VERDICT'].values
test_x = test_data.drop('Id', axis=1).values
# test_y = test_data['Id'].values
test_y = sampleSubmission['VERDICT'].values

smote = SMOTE()
train_x_smote, train_y_smote = smote.fit_resample(train_x, train_y)

mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=100)
mlp.fit(train_x_smote, train_y_smote)

# Make predictions on the test data
test_y_pred = mlp.predict(test_x)

# Evaluate the accuracy of the predictions
accuracy = accuracy_score(test_y, test_y_pred)
print(f'Test accuracy: {accuracy}')

df = pd.DataFrame({'VERDICTS': test_y_pred})

df.to_csv('predictions.csv', index=False)

# import the necessary libraries for dataset preparation, feature engineering, model training
import pandas as pd
from sklearn.model_selection import KFold

from DataHandling import data_setup, data_extraction_test, data_extraction_train
from Resampling import resample_data_train, allknn_data_test

# Import Training and Testing Data
train = pd.read_csv('train.csv')
print("Training Set:"% train.columns, train.shape, len(train))
test = pd.read_csv('test.csv')
print("Test Set:"% test.columns, test.shape, len(test))

# Percentage of Positive/Negative
print("Positive: ", train.label.value_counts()[0]/len(train)*100, "%")
print("Negative: ", train.label.value_counts()[1]/len(train)*100, "%")

#loop for cross validation
train_clean, test_clean = data_setup(train, test)

# Training Dataset resampling
# Refer to resampling.py and datahandling.py for more information
x_tfidf, y, xvalid_tfidf, valid_y = data_extraction_train(train_clean)
resample_data_train(x_tfidf, y, xvalid_tfidf, valid_y)

# Testing Dataset, this will return a matrix with the models predictions for each tweet
print("\nTESTING DATASET\n")
x_tfidf, y, xvalid_tfidf, valid_x = data_extraction_test(train_clean, test_clean)
allknn_data_test(x_tfidf, y, xvalid_tfidf, test)



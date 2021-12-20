from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, AllKNN
from imblearn.combine import SMOTEENN, SMOTETomek
from DataHandling import train_model, train_model_test
from sklearn import linear_model, svm
import pandas as pd

# The scores given are f1 scores
# f1 score is explained as 2*(Recall * Precision) / (Recall + Precision)
# recall is the ratio of correctly predicted positive observations to all observations in the class
# precision is the ratio of correctly predicted positive observations to the total predicted positive observations


# Resample_data_train uses multiple resampling methods and returns and f1 score. The f1 score determines which model
# gives a high score without over-fitting.
def resample_data_train(x_tfidf, y, xvalid_tfidf, valid_y):
    # Original Accuracy with Data Imbalance
    print("\nOriginal accuracy with data imbalance")
    accuracyORIGINAL = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
                                   x_tfidf, y, xvalid_tfidf, valid_y)
    print("LR Baseline, WordLevel TFIDF: ", accuracyORIGINAL)
    accuracyORIGINAL = train_model(svm.LinearSVC(), x_tfidf, y, xvalid_tfidf, valid_y)
    print("SVM Baseline, WordLevel TFIDF: ", accuracyORIGINAL)
    
    # Oversampling
    # Random Oversampling
    print("\nOversampling:")
    print("\nRandom Oversampling")
    ros = RandomOverSampler(random_state=777)
    ros_x_tfidf, ros_y = ros.fit_resample(x_tfidf, y)
    accuracyROS = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs', max_iter=200, multi_class='multinomial'),
                              ros_x_tfidf, ros_y, xvalid_tfidf, valid_y)
    print("LR ORIGINAL, WordLevel TFIDF: ", accuracyROS)
    accuracyROS = train_model(svm.LinearSVC(), ros_x_tfidf, ros_y, xvalid_tfidf, valid_y)
    print("SVM ROS, WordLevel TFIDF: ", accuracyROS)
    
    # SMOTE
    # SMOTE: Synthetic Minority Over-sampling Technique.”
    # SMOTE works by selecting examples that are close in the feature space,
    # drawing a line between the examples in the feature space and drawing a new
    # sample at a point along that line.
    print("\nSMOTE Oversampling")
    sm = SMOTE(random_state=777)
    sm_x_tfidf, sm_y = sm.fit_resample(x_tfidf, y)
    accuracySMOTE = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs', max_iter=200, multi_class='multinomial'),
                                sm_x_tfidf, sm_y, xvalid_tfidf, valid_y)
    print("LR SMOTE, WordLevel TFIDF: ", accuracySMOTE)
    accuracySMOTE = train_model(svm.LinearSVC(), sm_x_tfidf, sm_y, xvalid_tfidf, valid_y)
    print("SVC SMOTE, WordLevel TFIDF: ", accuracySMOTE)
    
    # Under-sampling
    # Random Under-sampling
    print("\nUnder-sampling:")
    print("\nRandom Under-sampling")
    rus = RandomUnderSampler(random_state=0, replacement=True)
    rus_x_tfidf, rus_y = rus.fit_resample(x_tfidf, y)
    accuracyrus = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs', max_iter=200, multi_class='multinomial'),
                              rus_x_tfidf, rus_y, xvalid_tfidf, valid_y)
    print("LR RUS, WordLevel TFIDF: ", accuracyrus)
    accuracyrus = train_model(svm.LinearSVC(), rus_x_tfidf, rus_y, xvalid_tfidf, valid_y)
    print("SVC RUS, WordLevel TFIDF: ", accuracyrus)
    
    # NearMiss Version 2
    # Near Miss refers to a collection of under-sampling methods that select examples based on the
    # distance of majority class examples to minority class examples
    print("\nNearMiss Version 2")
    nm = NearMiss(version=2)
    nm_x_tfidf, nm_y = nm.fit_resample(x_tfidf, y)
    accuracynm = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
                             nm_x_tfidf, nm_y, xvalid_tfidf, valid_y)
    print("LR NearMiss, WordLevel TFIDF: ", accuracynm)
    accuracynm = train_model(svm.LinearSVC(), nm_x_tfidf, nm_y, xvalid_tfidf, valid_y)
    print("SVC NearMiss, WordLevel TFIDF: ", accuracynm)
    
    # Combination of oversampling and under-sampling aka resampling
    # Re-Sampling SMOTEENN
    # Over-sampling using SMOTE and cleaning using ENN.
    # Combines over- and under-sampling using SMOTE and Edited Nearest Neighbours.
    print("\nResampling:")
    print("\nResampling SMOTE and Edited Nearest Neighbor")
    se = SMOTEENN(random_state=42)
    se_x_tfidf, se_y = se.fit_resample(x_tfidf, y)
    accuracyse = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),
                             se_x_tfidf, se_y, xvalid_tfidf, valid_y)
    print ("LR SMOTEENN: ", accuracyse)
    accuracyse = train_model(svm.LinearSVC(), se_x_tfidf, se_y, xvalid_tfidf, valid_y)
    print ("SVC SMOTEENN: ", accuracyse)
    
    # Re-Sampling SMOTETomek
    # SMOTETomek is a hybrid method which is a mixture of the above two methods,
    # it uses an under-sampling method (Tomek) with an oversampling method (SMOTE)
    print("\nResampling SMOTE and Tomek Link Removal")
    st = SMOTETomek(random_state=42)
    st_x_tfidf, st_y = st.fit_resample(x_tfidf, y)
    accuracyst = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
                             st_x_tfidf, st_y, xvalid_tfidf, valid_y)
    print("LR SMOTETomek: ", accuracyst)
    accuracyst = train_model(svm.LinearSVC(), st_x_tfidf, st_y, xvalid_tfidf, valid_y)
    print("SVC SMOTETomek: ", accuracyst)

    # AllKNN
    # AllKN is an under-sampling technique based on Edited Nearest Neighbors. These techniques try to
    # under-sample your majority classes by removing samples that are close to the minority class, in order to make your
    # classes more separable
    print("\nAllKNN")
    knn = AllKNN(allow_minority=True)
    knn_x_tfidf, knn_y = knn.fit_resample(x_tfidf, y)
    accuracyknn = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
                             knn_x_tfidf, knn_y, xvalid_tfidf, valid_y)
    print("LR AllKNN: ", accuracyknn)
    accuracyknn = train_model(svm.LinearSVC(), knn_x_tfidf, knn_y, xvalid_tfidf, valid_y)
    print("This is the highest performing resampling method.\nWe will use this for the testing data.")
    print("SVM AllKNN: ", accuracyknn)

# The All KNN was the best model for the training dataset so we will use it for the testing dataset
# AllKN is an under-sampling technique based on Edited Nearest Neighbors. These techniques try to
# under-sample your majority classes by removing samples that are close to the minority class, in order to make your
# classes more separable

def allknn_data_test(x_tfidf, y, xvalid_tfidf, test):
    #AllKNN
    print("\nAllKNN using testing data:")
    knn = AllKNN(allow_minority=True)
    knn_x_tfidf, knn_y = knn.fit_resample(x_tfidf, y)
    accuracyknn = train_model_test(svm.LinearSVC(), knn_x_tfidf, knn_y, xvalid_tfidf)
    print("SVM AllKNN: ", accuracyknn)
    print("Accuracy matrix is being loaded into a csv file named \"test-predictions-knn.csv\"")

    # Convert to CSV file
    d = {'id': test['id'], 'Tweet': test['tweet'], 'label': accuracyknn}
    df = pd.DataFrame(data=d)
    df.to_csv("test_predictions_knn.csv", index=False)

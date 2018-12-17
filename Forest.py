# data analysis and wrangling
import pandas as pd
# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.svm import SVC

# Read csv
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

# Complete Values
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# Convert the male and female groups to integer form
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
# Convert the Embarked classes to integer form
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
# Convert the male and female groups to integer form on test
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
# Convert the Embarked classes to integer form ontes
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

train.drop(['Name', 'Ticket'],axis=1, inplace=True)
test.drop(['Name', 'Ticket'],axis=1, inplace=True)
train.drop(labels=["Cabin"], axis=1, inplace=True)
test.drop(labels = ['Cabin'], axis=1, inplace=True)

'''Complete Age
age_train = KNN(k=10).complete(train)
age_test = KNN(k=10).complete(test)
train = pd.DataFrame(age_train, columns = train.columns)
test = pd.DataFrame(age_test, columns = test.columns)
'''

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
train["family_size"] = train["SibSp"] + train["Parch"] + 1
test["family_size"] = test["SibSp"] + test["Parch"] + 1
train["isAlone"] = 0
test["isAlone"] = 0
train.loc[train["family_size"] == 1, "isAlone"] = 1
test.loc[test["family_size"] == 1, "isAlone"] = 1
features_forest = train[["Pclass", "Age", "Sex", "Fare", "Embarked", "isAlone"]].values
test_features = test[["Pclass", "Age", "Sex", "Fare", "Embarked", "isAlone"]].values

# Scaling
test_features = preprocessing.scale(test_features)
features_forest = preprocessing.scale(features_forest)

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the feature_importances_
pred_forest = my_forest.predict(test_features)
print(my_forest.feature_importances_)

# Add PassengerId to Survived as a column
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns=["Survived"])
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("forest_solution.csv", index_label=["PassengerId"])

svc = SVC()
svc.fit(features_forest, target)
svc_pred = svc.predict(test_features)
print(svc.score(features_forest, target))

# Add PassengerId to Survived as a column
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(svc_pred, PassengerId, columns=["Survived"])
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("SVM_solution.csv", index_label=["PassengerId"])
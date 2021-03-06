# data analysis and wrangling
import pandas as pd
# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree

# Read csv
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
test_target = pd.read_csv("input/gender_submission.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
# Convert the male and female groups to integer form
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
# Convert the Embarked classes to integer form
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

# Impute the missing value with the median on test
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
# Convert the male and female groups to integer form on test
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
# Convert the Embarked classes to integer form ontes
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

# Create the target and features numpy arrays: target, features_one, test_features_one, test_target
target = train["Survived"].values
test_target = test_target["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
test_features_one = test[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Print the score of the new decison tree
print(my_tree_one.score(test_features_one, test_target))

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features_one)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])

# Write your solution to a csv file with the name my_solution.csv
# my_solution.to_csv("my_solution_one.csv", index_label=["PassengerId"])

# Create a new array with the added features: features_two
features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
test_features_two = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
my_tree_two = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=1)
my_tree_two = my_tree_two.fit(features_two, target)

# Print the score of the new decison tree
print(my_tree_two.score(test_features_two, test_target))

# Make your prediction using the test set
my_prediction2 = my_tree_two.predict(test_features_two)
# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution2 = pd.DataFrame(my_prediction2, PassengerId, columns=["Survived"])
# my_solution2.to_csv("my_solution_two.csv", index_label=["PassengerId"])

# Create a new feature set and remove Embarked
features_three = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]].values
test_features_three = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=1)
my_tree_three = my_tree_three.fit(features_three, target)

# Print the score of this decision tree
print(my_tree_three.score(test_features_three, test_target))

# Make your prediction using the test set
my_prediction3 = my_tree_three.predict(test_features_three)
# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution3 = pd.DataFrame(my_prediction3, PassengerId, columns=["Survived"])
my_solution3.to_csv("my_solution3.csv", index_label=["PassengerId"])

# Create train_two/test_two with the newly defined feature
train_two = train.copy()
train_two["FamilySize"] = train_two["SibSp"] + train_two["Parch"] + 1
test_two = test.copy()
test_two["FamilySize"] = test_two["SibSp"] + test_two["Parch"] + 1

# Create a new feature set and add the new feature
features_four = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "FamilySize"]].values
test_features_four = test_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "FamilySize"]].values

# Define the tree classifier, then fit the model
my_tree_four = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=1)
my_tree_four = my_tree_four.fit(features_four, target)

# Print the score of this decision tree
print(my_tree_four.score(features_four, target))

# Make your prediction using the test set
my_prediction4 = my_tree_four.predict(test_features_four)
# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution4 = pd.DataFrame(my_prediction3, PassengerId, columns=["Survived"])
my_solution4.to_csv("my_solution4.csv", index_label=["PassengerId"])
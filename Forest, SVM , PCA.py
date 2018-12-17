# data analysis and wrangling
import pandas as pd
# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

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
train_features = train[["Pclass", "Age", "Sex", "Fare", "Embarked", "isAlone"]].values
test_features = test[["Pclass", "Age", "Sex", "Fare", "Embarked", "isAlone"]].values

# Scaling
test_features = StandardScaler().fit_transform(test_features)
train_features = StandardScaler().fit_transform(train_features)

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
my_forest = forest.fit(train_features, target)

# Print the score of the fitted random forest
print(my_forest.score(train_features, target))

# Compute predictions on our test set features then print the feature_importances_
pred_forest = my_forest.predict(test_features)
print(my_forest.feature_importances_)

# Add PassengerId to Survived as a column
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns=["Survived"])
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("forest_solution.csv", index_label=["PassengerId"])


#PCA to visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_test = pca.fit_transform(test_features)
pca_train = pca.transform(train_features)

svc = SVC()
svc.fit(pca_train, target)
svc_pred = svc.predict(pca_test)
print(svc.score(pca_train, target))

# Add PassengerId to Survived as a column
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(svc_pred, PassengerId, columns=["Survived"])
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("SVM&PCA_solution.csv", index_label=["PassengerId"])

corr = train.corr()**2
print(corr["Survived"].sort_values(ascending=False))

'''
Show PCA
principalDf = pd.DataFrame(data = pca_test
             , columns = ['principal component 1', 'principal component 2'])
principalDf["Survived"] = svc_pred
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = principalDf["Survived"] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
'''
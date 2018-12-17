# data analysis and wrangling
import pandas as pd
# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn.cluster import KMeans
from sklearn import preprocessing
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

train.drop(['Name', 'Ticket'], axis=1, inplace=True)
test.drop(['Name', 'Ticket'], axis=1, inplace=True)
train.drop(labels=["Cabin"], axis=1, inplace=True)
test.drop(labels = ['Cabin'], axis=1, inplace=True)

train["family_size"] = train["SibSp"] + train["Parch"] + 1
test["family_size"] = test["SibSp"] + test["Parch"] + 1
train["isAlone"] = 0
test["isAlone"] = 0
train.loc[train["family_size"] == 1, "isAlone"] = 1
test.loc[test["family_size"] == 1, "isAlone"] = 1

X_train = train[["Pclass", "Age", "Sex", "Fare", "Embarked", "isAlone"]].values
X_test = test[["Pclass", "Age", "Sex", "Fare", "Embarked", "isAlone"]].values

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
y = np.array(train["Survived"])

'''
print(y.describe())
print(X.describe())
'''
# PCA on data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# KMeans
clf = KMeans(n_clusters=2)
clf.fit(X_train)
clf_pred = clf.predict(X_test)
correct = 0

#Verifica a % de acertos

for i in range(len(clf_pred)):
    if clf_pred[i]==y[i]:
        correct+=1
print(max(1 - correct / len(X_train), correct / len(X_train)))

clf_pred = clf.predict(X_test)
PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(clf_pred, PassengerId, columns=["Survived"])
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("KMeans.csv", index_label=["PassengerId"])

for i in range(len(clf_pred)) :
    if clf_pred[i] == 1:
        clf_pred[i] = 0
    else:
        clf_pred[i] = 1
my_solution = pd.DataFrame(clf_pred, PassengerId, columns=["Survived"])
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("KMeans_inv.csv", index_label=["PassengerId"])

'''
Plot Data
principalDf = pd.DataFrame(data = X_test
             , columns = ['principal component 1', 'principal component 2'])
principalDf["Survived"] = clf_pred
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
columns = train.keys().tolist()
for i in columns:
    count1 = train.loc[train['Survived'] == 1, i].value_counts(normalize=False)
    count0 = train.loc[train['Survived'] == 0, i].value_counts(normalize=False)
    values1 = train.loc[train['Survived'] == 1, i].value_counts().keys().tolist()
    values0 = train.loc[train['Survived'] == 0, i].value_counts().keys().tolist()
    data0 = pd.DataFrame(count0, values0)
    data1 = pd.DataFrame(count1, values1)
    data0.sort_index(inplace=True)
    data1.sort_index(inplace=True)
    print(data0)
    print(data1)
    print(i)
    y_pos0 = np.arange(len(values0))
    y_pos1 = np.arange(len(values1))
    plt.bar(y_pos0*2, data0[i], align='center', label='Cluster0', color='b')
    plt.bar(y_pos1*2+1, data1[i], align='center', label='Cluster1', color='r')
    plt.xticks(y_pos0*2, data0.index)
    plt.title(i)
    plt.legend()
    plt.show()

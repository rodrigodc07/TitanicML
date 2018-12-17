import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

train.Embarked.fillna('S', inplace=True)
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
test.drop(labels=['Cabin'], axis=1, inplace=True)



from fancyimpute import KNN
age_train = KNN(k=10).complete(train)
age_test = KNN(k=10).complete(test)

train = pd.DataFrame(age_train, columns = train.columns)
test = pd.DataFrame(age_test, columns = test.columns)
train["family_size"] = train["SibSp"] + train["Parch"] + 1
test["family_size"] = test["SibSp"] + test["Parch"] + 1
train["isAlone"] = 0
test["isAlone"] = 0
train.loc[train["family_size"] == 1, "isAlone"] = 1
test.loc[test["family_size"] == 1, "isAlone"] = 1

# Separating our independent and dependent variable
X = train[["Pclass", "Age", "Sex", "Fare", "isAlone"]]
y = train["Survived"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = .33, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
svc_accy = round(accuracy_score(svc_pred, y_test), 3)
print("Acerto do SVM ", svc_accy)

clf = KMeans(n_clusters=2)
clf.fit(x_train)
clf_pred = clf.predict(x_test)
svc_accy = round(accuracy_score(clf_pred, y_test), 3)
svc_accy= (max(1-svc_accy, svc_accy))
print("Acerto do K-Means ",svc_accy)


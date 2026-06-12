import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns 
import numpy as np

# #Iris Classification

iris = sns.load_dataset("iris")
print(iris.head())
print(iris.info())
print(iris.isnull().sum())

#EDA

print(iris.shape)
print(iris.columns)
print(iris.duplicated().sum()) 

#Statitical Measures

print(iris.describe())
print(iris['species'].value_counts())

#Box Plot [To detect Outliers: ]

sns.boxplot(x= iris['sepal_length'])
plt.xlabel("Sepal Length")
plt.ylabel("Values")
plt.show()

sns.boxplot(x= iris['petal_length'])
plt.xlabel("Petal Length")
plt.ylabel("Values")
plt.show()

sns.boxplot(x= iris['petal_width'])
plt.xlabel("Petal Width")
plt.ylabel("Values")
plt.show()

sns.boxplot(x= iris['sepal_width'])
plt.xlabel("Sepal Width")
plt.ylabel("Values")
plt.show()

# Correlation Heatmap

corr = iris.drop('species', axis=1).corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Correlation Heatmap of Iris Features")
plt.show()

#Pair Plot

sns.pairplot(iris, hue='species')
plt.show()


# Machine Learning Models

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Initialize train & test split

X = iris.drop('species', axis=1)
y = iris['species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 42)


#logistic Regression

Model = LogisticRegression()
Tran = Model.fit(X_train,y_train)
Pred = Model.predict(X_test)
print(Pred) 
print(y_test)
print("Classification Report: ")
print(classification_report(y_test, Pred))
print("Confusion Matrix: ",confusion_matrix(y_test,Pred))


Model1 = DecisionTreeClassifier()
Model1.fit(X_train,y_train)
Pred1 = Model1.predict(X_test)
print("Decision Treee Prediction: ",Pred1)
print("Classification Report: ")
print(classification_report(y_test,Pred1))
print("Confusion Matric [DCT]: ")
print(confusion_matrix(y_test,Pred1))


from sklearn.neighbors import KNeighborsClassifier
Model2 = KNeighborsClassifier()
Model2.fit(X_train,y_train)
Pred2 = Model2.predict(X_test)
print("Classification Report: ")
print(classification_report(y_test,Pred2))
print("Confusion Matrix [KNN]")
print(confusion_matrix(y_test,Pred2))

#Accuracy

log_acc = accuracy_score(y_test, Pred)
dt_acc = accuracy_score(y_test, Pred1)
knn_acc = accuracy_score(y_test, Pred2)

print("Accuracy Scores: ")

comparison = pd.DataFrame({'Model': ['Logistic Regression', 'Decision Tree', 'KNN'],'Accuracy': [log_acc, dt_acc, knn_acc]})

print(comparison)


#The End [Ashraf Bari Abdul Ahad Abdul Bari]
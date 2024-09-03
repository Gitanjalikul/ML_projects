'''in this project we are going to predict sales using logistic regression and KNN
 and will compare the performamce by both algorithms'''

#importing essential libraries
import pandas as pd
import numpy as np

#load dataset from local directory
dataset = pd.read_csv("E:\my_programs\python_programs\ML\DigitalAd_dataset.csv")


#briefing dataset
print('printing dataset shape', dataset.shape)
print('first few lines of dataset\n', dataset.head())
#checking for null values
print(dataset.isnull().sum())

#define dependent and independant variable
x = dataset.iloc[: , :-1].values

y = dataset.iloc[ : , -1].values

#spliting dataset into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


#transform data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train) #fit transform calculates mean and variance for each feature and transforms he data
X_test = sc.transform(x_test) #transform the data based upon training data

#model training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#evaluating model by confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)*100
print("confusion matrix for logistic regression:\n", cm)
print('Accuracy score by logistic regression model:', acc_score)

#estimation of ROC curve and AUC for logistic regression

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#predict probabilities for positive class
y_prob = model.predict_proba(X_test)[: , 1]

fpr, tpr, threshold = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color = 'blue', label = 'ROC curve (area = %0.2f)' %roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

print(f'AUC Score: {roc_auc:.2f}')

'''
sale prediction using KNN algorithm
'''
#finding best value for k
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
k_range = range(1,20)
error = []
for k in k_range:
    mod_knn = KNeighborsClassifier(n_neighbors=k)
    mod_knn.fit(X_train, y_train)
    y_pred_knn=mod_knn.predict(X_test)
    error.append(np.mean(y_pred_knn != y_test))

plt.plot(k_range, error, marker = 'o', color = 'blue')
plt.xticks(ticks= np.arange(1,20))
plt.title('k value graph')
plt.xlabel('k value')
plt.ylabel('error')
plt.show()

#training
model1 = KNeighborsClassifier(n_neighbors=3)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(x_test)

#evaluating model by confusion matrix
cm1 = confusion_matrix(y_test, y_pred1)
print('confusion matrix for KNN:\n', cm1)
acc_score1 = accuracy_score(y_test, y_pred1)*100
print('accuracy score for KNN model is:\n', acc_score1)

#prediction for new customer
try:
    age = int(input('Enter age\n:'))
    salary = int(input('Enter salary:\n'))
    var = sc.transform([[age, salary]])
    result = model.predict(var)
    if result ==1:
        print('customer will buy product')
    else:
        print('customer will not buy product')

except ValueError:
        print('Please enter valid numerical values for age and salary.')


# load libraries
from pandas import read_csv
from csv import reader
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

filename = 'raw-data-intext-training.data'

names = ['age', 'menopause', 'tumor-size', 'inv-nodes']

dataset = read_csv(filename, names=names, delimiter=',')


# spilt-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 3]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Spot-Check Algorithms
models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC())]

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


#LR: 0.991304 (0.026087)
#LDA: 0.426482 (0.113194)
#KNN: 0.635771 (0.073650)
#CART: 1.000000 (0.000000)
#NB: 1.000000 (0.000000)
#SVM: 0.448617 (0.109097)
#
#0.6379310344827587
#[[ 5  6  0]
# [ 4 24  5]
# [ 0  6  8]]
#              precision    recall  f1-score   support
#
#         1.0       0.56      0.45      0.50        11
#         2.0       0.67      0.73      0.70        33
#         3.0       0.62      0.57      0.59        14
#
#    accuracy                           0.64        58
#   macro avg       0.61      0.58      0.60        58
#weighted avg       0.63      0.64      0.63        58

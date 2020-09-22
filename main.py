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

filename = 'raw-data-intext.data'
names = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node_caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat',
         'class']
dataset = read_csv(filename, names=names, delimiter=',')

print(dataset.shape)
print(dataset.head(20))

print(dataset.describe(include='all'))

print(dataset.groupby('class').size())

dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

dataset.hist()
pyplot.show()

scatter_matrix(dataset)
pyplot.show()

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

#(286, 10)
#      age menopause tumor-size  ... breast-quad irradiat                 class
#0   30-39   premeno      30-34  ...    left_low       no  no-recurrence-events
#1   40-49   premeno      20-24  ...    right_up       no  no-recurrence-events
#2   40-49   premeno      20-24  ...    left_low       no  no-recurrence-events
#3   60-69      ge40      15-19  ...     left_up       no  no-recurrence-events
#4   40-49   premeno        0-4  ...   right_low       no  no-recurrence-events
#5   60-69      ge40      15-19  ...    left_low       no  no-recurrence-events
#6   50-59   premeno      25-29  ...    left_low       no  no-recurrence-events
#7   60-69      ge40      20-24  ...    left_low       no  no-recurrence-events
#8   40-49   premeno      50-54  ...    left_low       no  no-recurrence-events
#9   40-49   premeno      20-24  ...     left_up       no  no-recurrence-events
#10  40-49   premeno        0-4  ...     central       no  no-recurrence-events
#11  50-59      ge40      25-29  ...    left_low       no  no-recurrence-events
#12  60-69      lt40      10-14  ...    right_up       no  no-recurrence-events
#13  50-59      ge40      25-29  ...    right_up       no  no-recurrence-events
#14  40-49   premeno      30-34  ...     left_up       no  no-recurrence-events
#15  60-69      lt40      30-34  ...    left_low       no  no-recurrence-events
#16  40-49   premeno      15-19  ...    left_low       no  no-recurrence-events
#17  50-59   premeno      30-34  ...    left_low       no  no-recurrence-events
#18  60-69      ge40      30-34  ...    left_low       no  no-recurrence-events
#19  50-59      ge40      30-34  ...    right_up       no  no-recurrence-events
#
#[20 rows x 10 columns]
#          age menopause tumor-size  ... breast-quad irradiat                 class
#count     286       286        286  ...         286      286                   286
#unique      6         3         11  ...           6        2                     2
#top     50-59   premeno      30-34  ...    left_low       no  no-recurrence-events
#freq       96       150         60  ...         110      218                   201
#mean      NaN       NaN        NaN  ...         NaN      NaN                   NaN
#std       NaN       NaN        NaN  ...         NaN      NaN                   NaN
#min       NaN       NaN        NaN  ...         NaN      NaN                   NaN
#25%       NaN       NaN        NaN  ...         NaN      NaN                   NaN
#50%       NaN       NaN        NaN  ...         NaN      NaN                   NaN
#75%       NaN       NaN        NaN  ...         NaN      NaN                   NaN
#max       NaN       NaN        NaN  ...         NaN      NaN                   NaN
#
#[11 rows x 10 columns]
#class
#no-recurrence-events    201
#recurrence-events        85
#dtype: int64


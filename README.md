# Implementation of K-Means Clustering Algorithm
## Aim
To write a python program to implement K-Means Clustering Algorithm.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation

## Algorithm:

### Step1

import pandas as pd
### Step2

we the read the file using pandas.

### Step3

we print first two lines using pandas.

### Step4

we give 2 variales as X1 and X2.

### Step5

print clusters and centre.

## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
 
X1 = pd.read_csv('clustering.csv')
print(X1.head(2))
X2 = X1.loc[: , ['ApplicantIncome', 'LoanAmount']]
print(X2.head(2))

X = X2.values 
sns.scatterplot(X[:,0], X[:, 1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()

kmean=KMeans(n_clusters=4)
kmean.fit(X)


print('Cluster Centers:',kmean.cluster_centers_)
print('Labels:', kmean.labels_)
 
predicted_class = kmean.predict([[9000,120]])
print('The cluster group for Application Income 9000 and Loanamount:',predicted_class)





```
## Output:
![output](.//img1.png)
![output](.//img2.png)

### Insert your output

<br>

## Result
Thus the K-means clustering algorithm is implemented and predicted the cluster class using python program.

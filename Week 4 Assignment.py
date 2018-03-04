# -*- coding: utf-8 -*-
"""
Machine Learning for Data Analysis
Week 4 Assignment
K Means Clustering
Written by: Mike Silva
"""

# Import libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
sns.set_style('whitegrid')
sns.set_context('talk')

# Eliminate false positive SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Make results reproducible
np.random.seed(1234567890)

df = pd.read_csv('gapminder.csv')

variables = ['incomeperperson', 'alcconsumption', 'co2emissions', 'femaleemployrate', 
                'internetuserate', 'lifeexpectancy','polityscore','employrate','urbanrate'] 

# convert to numeric format
for variable in variables:
    df[variable] = pd.to_numeric(df[variable], errors='coerce')

# listwise deletion of missing values
subset = df[variables].dropna()

# Print the rows and columns of the data frame
print('Size of study data')
print(subset.shape)
print("\n")
"""
"===============  Random Forest to Select Clustering Variables  ===============
"""
n_estimators=25

subset['incomequartiles'] = pd.cut(subset['incomeperperson'], 3, labels=['0%-33%','34%-66%','67%-100%'])
subset['incomequartiles'] = subset['incomequartiles'].astype('category')

variables.pop(0)

predictors = subset[variables]
targets = subset['incomequartiles']

#Split into training and testing sets+
training_data, test_data, training_target, test_target  = train_test_split(predictors, targets, test_size=.25)

# Build the random forest classifier
classifier=RandomForestClassifier(n_estimators=n_estimators)
classifier=classifier.fit(training_data,training_target)

predictions=classifier.predict(test_data)

# Fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(training_data,training_target)

# Display the relative importance of each attribute
feature_name = list(predictors.columns.values)
feature_importance = list(model.feature_importances_)
features = pd.DataFrame({'name':feature_name, 'importance':feature_importance}).sort_values(by='importance', ascending=False)
print(features.head(len(feature_name)))

"""
" =============================  Data Management  =============================
"""
variables = ['incomeperperson', 'lifeexpectancy', 'internetuserate', 'urbanrate'] 

# convert to numeric format
for variable in variables:
    df[variable] = pd.to_numeric(df[variable], errors='coerce')

# listwise deletion of missing values
subset = df[variables].dropna()

# Print the rows and columns of the data frame
print('Size of study data')
print(subset.shape)

subset['incomequartiles'] = pd.cut(subset['incomeperperson'], 3, labels=['0%-33%','34%-66%','67%-100%'])
subset['incomequartiles'] = subset['incomequartiles'].astype('category')

# Remove the first variable from the list since the target is derived from it
variables.pop(0)

# Center and scale data
for variable in variables:
    subset[variable]=preprocessing.scale(subset[variable].astype('float64'))
    
features = subset[variables]
targets = subset[['incomeperperson']]

"""
" ==================  Split Data into Training and Test Sets  ==================
"""
# Split into training and testing sets
training_data, test_data, training_target, test_target  = train_test_split(features, targets, test_size=.3)
print('Size of training data')
print(training_data.shape)

"""
" =====================  Determine the Number of Clusters  ====================
"""
# Identify number of clusters using the elbow method
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(training_data)
    clusassign=model.predict(training_data)
    meandist.append(sum(np.min(cdist(training_data, model.cluster_centers_, 'euclidean'), axis=1)) / training_data.shape[0])

# Visualize the elbow
k = 2

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(clusters, meandist)
ax.plot(clusters[(k-1)], meandist[(k-1)], marker='o', markersize=12, 
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of Clusters')
plt.ylabel('Average Distance')
plt.title('Selecting K with the Elbow Method')
plt.show()

"""
" ==========================  Visualize the Clusters  =========================
"""
model=KMeans(n_clusters=k)
model.fit(training_data)
training_data['cluster'] = model.labels_

my_cmap = plt.cm.get_cmap('brg')
my_cmap.set_under('w')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(training_data.iloc[:,0], training_data.iloc[:,1], training_data.iloc[:,2], c=training_data['cluster'], cmap=my_cmap)
ax.set_xlabel(training_data.columns.values[0])
ax.set_ylabel(training_data.columns.values[1])
ax.set_zlabel(training_data.columns.values[2])
plt.show()
 
sns.pairplot(training_data, hue ='cluster');

"""
" ====================  Examine Differences Between Clusters  =================
"""

training_target['cluster'] = model.labels_
income_model = smf.ols(formula='incomeperperson ~ C(cluster)', data=training_target).fit()
print (income_model.summary())

print ('means for features by cluster')
m1= training_target.groupby('cluster').mean()
print (m1)

print ('standard deviations for features by cluster')
m2= training_target.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(training_target['incomeperperson'], training_target['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())

Decision Tree Classifier - Bank Marketing Dataset

Step 1: Load and Explore the Dataset

Download the dataset from the UCI Machine Learning Repository.

```python

import pandas as pd

data = pd.read_csv('bank-full.csv', sep=';')

print(data.info())

print(data.head())

```

Step 2: Preprocess the Data

Encode categorical variables and the target variable.

```python

from sklearn.preprocessing import LabelEncoder

data['y'] = data['y'].map({'yes': 1, 'no': 0})

categorical_cols = data.select_dtypes(include=['object']).columns

label_encoders = {}

for col in categorical_cols:

 le = LabelEncoder()

 data[col] = le.fit_transform(data[col])

 label_encoders[col] = le
Step 3: Train a Decision Tree Classifier

Split the data and train the model.

```python

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

X = data.drop('y', axis=1)

y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

```

Step 4: Visualize the Decision Tree

Visualize the trained decision tree.

```python

from sklearn import tree

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)

plt.show()

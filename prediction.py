import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

trainpath = './traindf-2.csv'
traindf = pd.read_csv(trainpath, delimiter=",")

testpath = './testdf-2.csv'
testdf = pd.read_csv(testpath, delimiter=",")

### model

y = traindf["Survived"]

X = traindf.drop(['Survived'], axis=1)
X_test = testdf

model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': testdf.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

// https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
// https://www.kaggle.com/uciml/pima-indians-diabetes-database

if[not `p in key `;system "l p.q"]

np:.p.import`numpy
xgbClassifier:.p.import[`xgboost;`:XGBClassifier]

// Accuracy score from sklearn
accuracy_score:.p.import[`sklearn.metrics;`:accuracy_score;<]

// Alternative accuracy score in q
accuracy_score_q:{sum[x=y]%count x}

// sklearn train_test_split in q
train_test_split:{[x;y;sz]`xtrain`ytrain`xtest`ytest!raze(x;y)@\:/:(0,floor n*1-sz)_neg[n]?n:count x}

// Define rounding function in Python and map to kdb+
p)def round(x): return "{:.2f}".format(x)
round:.p.get[`round;<]

// Util for converting kdb+ list to numpy array
kdb2np:{np[`:array][x]}


// Load data
dataset:flip ("IIIIIFFJH";",")0:`$":C:/q/w64/pima-natives-diabetes.csv"

// Split data into X and y
X:-1_'dataset
y:dataset[;-1+count dataset[0]]

// Split data into train and test sets
data:kdb2np each train_test_split[X;y;0.33]

// Instantiate classifier object from class
xgbModel:xgbClassifier[]

// Fit model, takes numpy arrays as input
fittedModel:xgbModel[`:fit][data`xtrain;data`ytrain]

// Obtain predictions based on test data
predictions:fittedModel[`:predict][data`xtest]`

// Calculate accuracy score
accuracy:accuracy_score[data[`ytest]`;predictions]

// Print rounded accuracy score
"Accuracy: ", round[accuracy * 100f], "%"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.datasets import load_diabetes
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


#load dataset 
diabetes = load_diabetes()
X = diabetes['data']
y = diabetes['target']
#print and assign feature names 
feature_names = diabetes.feature_names
X_df = pd.DataFrame(X, columns=feature_names)
y_df = pd.DataFrame(y, columns=['target'])
print(X_df.head())

#One Hot Encoding
nominal_column = ['sex']
enc = preprocessing.OneHotEncoder(categories='auto')
df_hd_named_enc = pd.DataFrame(enc.fit_transform(
    X_df[nominal_column]).toarray())
df_hd_named_enc.columns = enc.get_feature_names_out(nominal_column)
df_ohe = pd.concat([X_df, df_hd_named_enc], axis=1)
df_ohe.head(10)

#split 80/20
X_train, X_test, y_train, y_test = train_test_split(df_ohe, y, test_size=0.2, random_state=42, shuffle=True)

#hyperparameter tuning
param_grid = [
    {'kernel': ['linear', 'rbf', 'sigmoid']}, 
    {'kernel': ['poly'], 'degree': [1, 2, 3, 4, 5]}  # Degree of polynomial kernel
]
#GridSearchCV 
gs = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gs.fit(X_train, y_train)
#results
print( gs.best_params_)
print("Best CV MSE:", -gs.best_score_) #best degree=5 and poly 

#best parameters from GridSearchCV
svm_model = SVR(kernel='poly', degree=5) 
# Train the model
svm_model.fit(X_train, y_train)
# predictions
y_pred = svm_model.predict(X_test)

#checking shape of testdata
print("Shape of y_test:", y_test.shape)
print("Shape of y_pred:", y_pred.shape)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")


plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.title("Predictions vs. Actual Values")
plt.show()



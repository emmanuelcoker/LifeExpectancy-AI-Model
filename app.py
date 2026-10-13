import numpy as np # library for scientific and numerical computing
import pandas as pd # library for data wrangling, preparation and analysis
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


#read data from excel file
data_frame = pd.read_csv('life_expectancy.csv')
# print(data_frame.head())
# print(data_frame.info())
# print(data_frame.shape)#(2307, 11)

# check for missing values
# print(data_frame.isnull().sum())


#using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

#columns to encode
categorical_cols = ["Country Name", "Country Code", "Year"]

# Initialize encoder
enc = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

# Fit and transform categorical columns
encoded_data = enc.fit_transform(data_frame[categorical_cols])

# Convert encoded array to a DataFrame with column names
encoded_df = pd.DataFrame(
    encoded_data,
    columns=enc.get_feature_names_out(categorical_cols),
    index=data_frame.index
)

# Drop original categorical columns and join encoded columns
data_frame = pd.concat(
    [data_frame.drop(columns=categorical_cols), encoded_df],
    axis=1
)

# data_frame.head()


#set x(features) and y(target)
X = data_frame.drop('Life_Expectancy', axis = 1)
y = data_frame['Life_Expectancy']

#check features without the target
# X.head()

#data splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

#train the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully")

#Model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred) # 0 , 1

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

#export model and encoder
joblib.dump(enc, 'one_hot_encoder.pkl')
joblib.dump(model, 'life_expectancy_model.joblib')
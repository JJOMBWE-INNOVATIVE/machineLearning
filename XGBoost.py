import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, make_scorer

# Read the data
home_file_path = "C:/Users/isma/Desktop/ml/melb_data.csv"
home_data = pd.read_csv(home_file_path)

# Separate target from predictors
y = home_data.Price
# X = home_data.drop(['Price'], axis=1)
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = home_data[cols_to_use]

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)


# XGBOOST
from sklearn.metrics import mean_absolute_error, make_scorer
import xgboost as xgb

xgb_native_model = xgb.XGBRegressor(objective="reg:squarederror",
 n_estimators=1000,learning_rate=0.05,n_jobs=4)

# Convert data to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# Train the model with early stopping
params = {"objective":"reg:squarederror", "max_depth":3}
xgb_regressor = xgb.train(params, dtrain, num_boost_round=1000, 
    early_stopping_rounds=5,
    evals=[(dvalid, "eval")],
    verbose_eval=False
 )

# Make predictions using the trained model
dvalid_pred = xgb.DMatrix(X_valid)
predictions = xgb_regressor.predict(dvalid_pred)

print("Mean Absolute Error: " + str(mean_absolute_error
(predictions, y_valid)))
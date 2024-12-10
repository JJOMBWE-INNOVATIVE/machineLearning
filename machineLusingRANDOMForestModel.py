# [21:05, 12/8/2024] Jjombwe Nathan: # STEP BY STEP FOR MODELING
import pandas as pd
home_file_path = "C:/Users/Admin/Desktop/RESEARCH TRANSLATION/Housing.csv"
home_data = pd.read_csv(home_file_path)

# Filter rows with missing values
filtered_home_data = home_data.dropna()
# Drop rows with any missing values (default)
# filtered_home_data = home_data.dropna(axis=0, how='any')
# Drop rows with all missing values
# filtered_home_data = home_data.dropna(axis=0, how='all')

# print the list of columns in the dataset to find the name of the prediction target
# print(home_data.columns)

y = home_data.price
# print(y.describe())
# print(y.head())

# # Create the list of features below
feature_names = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking' ]

# # Select data corresponding to features in feature_names
X = home_data[feature_names]

# # Review data
# # print description or statistics from X
# print(X.describe())

# # print the top few lines
# print(X.head())

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# #specify the model. 
# #For model reproducibility, set a numeric value for random_state when specifying the model
home_model = DecisionTreeRegressor(random_state=1)

# # Fit the model
print(home_model.fit(X,y))

# # MAKE PREDICTIONS
print("First in-sample predictions:", home_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# # You can write code in this cell
# print(y.head())

# [22:55, 12/8/2024] Jjombwe Nathan: # MODEL MEAN ABSOLUTE ERROR

from sklearn.metrics import mean_absolute_error
predicted_home_prices = home_model.predict(X)
print('mean absolute error value for in- sample')
print(mean_absolute_error(y,predicted_home_prices))


# # VALIDATING DATA BY SPLITING IT 
from sklearn.model_selection import train_test_split

# # split data into training and validation data, for both features and target
# # The split is based on a random number generator. Supplying a numeric value to
# # the random_state argument guarantees we get the same split every time we
# # run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# # Define model
home_model = RandomForestRegressor(random_state=1)
# # Fit model
home_model.fit(train_X, train_y)

# Predict with all validation observations
print("Second out-sample predictions:",home_model.predict(val_X.head()))

# mean aboslute error validation data
print('validated data mean abosolute error after split, out - sample ')
val_predictions = home_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# compare MAE with differing values of max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

candidate_max_leaf_nodes = [5, 50, 500, 5000]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

# Print the scores for each leaf size
for leaf_size, mae in scores.items():
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(leaf_size, mae))

# Store the best value of max_leaf_nodes
best_tree_size = min(scores, key=scores.get)

# Fit Model Using All Data
final_model = RandomForestRegressor(max_leaf_nodes=best_tree_size, random_state=1)
print('best value for max leaf nodes')
# fit the final model and uncomment the next two lines
final_model.fit(X, y)
print(final_model)
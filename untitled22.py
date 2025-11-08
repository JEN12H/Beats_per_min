# -*- coding: utf-8 -*


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

# Commented out IPython magic to ensure Python compatibility.
# %pip install optuna

train_df= pd.read_csv('/content/train.csv')
test_df = pd.read_csv('/content/test.csv')

id = pd.DataFrame(train_df['id'])
print(id)
train_df = train_df.drop(['id'],axis = 1)
train_df.head()

train_df.shape

train_df.info()
# FOUND NO NULL VALUES

train_df.describe()

"""# PLOTTING THE DISTRIBUTION OF THE COLUMNS"""

print(train_df.columns)

plt.hist(x=train_df['RhythmScore'],bins=[0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],color='Blue')
plt.show()

plt.hist(x=train_df['AudioLoudness'],bins=[-25,-20,-15,-10,-5,0],color='Red')
plt.show()

plt.hist(x=train_df['Energy'],color='Green')
plt.show()

plt.hist(train_df['InstrumentalScore'],bins=[0,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7],color='Orange')
plt.show()

plt.hist(x=train_df['TrackDurationMs'],bins=[50000,100000,150000,200000,250000,300000,350000,400000,450000,500000],color='Yellow')
plt.xlabel('TrackDurationMs')
plt.ylabel('Frequency')
plt.title('Distribution of TrackDurationMs')
plt.show()

plt.hist(x=train_df['VocalContent'],color='Black')
plt.show()

plt.hist(train_df['LivePerformanceLikelihood'],color='Purple')
plt.show()

plt.hist(x=train_df['MoodScore'],color='Brown')
plt.show()

plt.hist(x=train_df['BeatsPerMinute'],color='Grey')
plt.show()

plt.hist(x=train_df['AcousticQuality'],color='Pink')
plt.show()

# SEPARATING COLUMNS BASED ON THE DISTRIBUTION
skewed_cols = ['AudioLoudness','AcousticQuality','LivePerformanceLikelihood','VocalContent','InstrumentalScore']
# REMAINING ARE UNIFORMLY DISTRIBUTED

for col in skewed_cols:
  sns.boxplot(x = col , data = train_df)
  plt.title(f"OUTLIERS IN {col}")
  print('\n')
  plt.show()

def handle_outlier(col):
  new_df = train_df.copy()
  for i in col:
    percentile25 = new_df[i].quantile(0.25)
    percentile75 = new_df[i].quantile(0.75)
    IQR = percentile75 - percentile25
    print('-'*50)
    print(f" THE IQR FOR {i} is {IQR}" )
    upper_limit = percentile75 + 1.5*IQR
    lower_limit = percentile25 - 1.5*IQR
    print(f" THE UPPER LIMIT FOR {i} IS {upper_limit}")
    print(f" THE LOWER LIMIT FOR {i} IS {lower_limit}")

    # Create a figure with two subplots for before and after outlier handling
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(f'Distribution and Box Plot for {i} Before and After Outlier Handling')

    # Before Outlier Handling
    sns.histplot(train_df[i], ax=axes[0, 0], kde=True)
    axes[0, 0].set_title('Before Outlier Handling (Distribution)')
    sns.boxplot(x=train_df[i], ax=axes[0, 1])
    axes[0, 1].set_title('Before Outlier Handling (Box Plot)')


    new_df[i] = np.where(new_df[i]> upper_limit,upper_limit,
                          np.where(new_df[i]<lower_limit,lower_limit,new_df[i]))

    print(" AFTER OUTLIER REMOVAL ")

    # After Outlier Handling
    sns.histplot(new_df[i], ax=axes[1, 0], kde=True)
    axes[1, 0].set_title('After Outlier Handling (Distribution)')
    sns.boxplot(x=new_df[i], ax=axes[1, 1])
    axes[1, 1].set_title('After Outlier Handling (Box Plot)')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

  return new_df

train_df = handle_outlier(skewed_cols)

train_df['TrackDurationMs'] = np.ceil(train_df['TrackDurationMs'] // 60000)
test_df['TrackDurationMs'] = np.ceil(test_df['TrackDurationMs'] // 60000)

X = train_df.drop(columns =['BeatsPerMinute'],axis = 1 )
y = train_df['BeatsPerMinute']

"""# PARAMETER TUNNING USING OPTUNA"""

def rmse(y_true,y_pred):
  return np.sqrt(np.mean((y_true-y_pred)**2))

rmse_scorer = make_scorer(rmse,greater_is_better=False)
# THIS MAKE_SCORER IS USED TO GET scorer object so that we can set greater value = False because of which
# sklearn minimizes the loss instead of maximizing accuracy.
def objective(trial):
  pt = PowerTransformer()
  scaler = StandardScaler()

  model_name = trial.suggest_categorical('model',['linear','rf','xgboost','lightgbm'])

  if model_name == 'linear':
    model = LinearRegression()
  elif model_name == 'rf':
    n_estimators = trial.suggest_int('rf_estimators',50,250) # Numer of trees in forest
    max_depth = trial.suggest_int('rf_max_depth',3,15) # Maximum depth of each decision tree
    min_samples_split = trial.suggest_int('rf_min_samples_split',2,10) # Minimum number of samples required to split and internal node
    model = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split, random_state = 42)

  elif model_name == 'xgboost':
    n_estimators = trial.suggest_int('xgb_estimators',50,250)
    max_depth = trial.suggest_int('xgb_max_depth',3,15)
    learning_rate = trial.suggest_float('xgb_learning_rate',0.001,1.0)
    model = XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,random_state=42)

  elif model_name =='lightgbm':
    n_estimators = trial.suggest_int('lgbm_estimators',50,250)
    max_depth = trial.suggest_int('lgbm_max_depth',3,15)
    learning_rate = trial.suggest_float('lgbm_learning_rate',0.001,1.0)
    model = LGBMRegressor(n_estimators=n_estimators,max_depth=max_depth,learning_rate = learning_rate,random_state=42)

  pipeline = Pipeline([
      ('powertransform',pt),
      ('standard_scaler',scaler),
      ('model', model)
  ])

  scores = cross_val_score(
      pipeline,
      X,
      y,
      cv = 3,
      scoring = rmse_scorer
  )

  mean_rmse = -np.mean(scores)
  return mean_rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective,n_trials=10,n_jobs = -1)

print("✅ Best model and params:")
print(study.best_trial.params)
best_params = study.best_trial.params
print(best_params['model'])
model_name = best_params['model']
print("Best RMSE:",study.best_value)

import pickle

# Create the best model based on the best parameters from Optuna
best_model_name = best_params['model']

if best_model_name == 'linear':
  best_model = LinearRegression()
elif best_model_name == 'rf':
  best_model = RandomForestRegressor(n_estimators=best_params['rf_estimators'],
                                    max_depth=best_params['rf_max_depth'],
                                    min_samples_split=best_params['rf_min_samples_split'],
                                    random_state=42)
elif best_model_name == 'xgboost':
  best_model = XGBRegressor(n_estimators=best_params['xgb_estimators'],
                            max_depth=best_params['xgb_max_depth'],
                            learning_rate=best_params['xgb_learning_rate'],
                            random_state=42)
elif best_model_name == 'lightgbm':
  best_model = LGBMRegressor(n_estimators=best_params['lgbm_estimators'],
                            max_depth=best_params['lgbm_max_depth'],
                            learning_rate=best_params['lgbm_learning_rate'],
                            random_state=42)

# Create a pipeline with the best model
best_pipeline = Pipeline([
    ('powertransform', PowerTransformer()),
    ('standard_scaler', StandardScaler()),
    ('model', best_model)
])

# Train the best model on the entire training data
best_pipeline.fit(X, y)

# Save the best model using pickle
with open('best_model.pkl', 'wb') as f:
  pickle.dump(best_pipeline, f)

print("✅ Best model saved as best_model.pkl")



# Load the saved model
with open('best_model.pkl', 'rb') as f:
  loaded_model = pickle.load(f)

# Make predictions on the test data
# Assuming 'id' column was dropped from test_df earlier, if not, drop it before predicting
test_predictions = loaded_model.predict(test_df.drop(columns=['id'], axis=1))

# Create a submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': test_predictions})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("✅ Predictions made and saved to submission.csv")












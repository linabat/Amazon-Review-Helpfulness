#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[2]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model

import json
import pandas as pd
import plotly.express as px

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# In[3]:


f = gzip.open("/Users/linabattikha/cse158/assignment2/data/reviews_Electronics_5.json.gz")
data = [] 
for l in f: 
    data.append(json.loads(l))


# In[5]:


raw_df = pd.DataFrame(data)


# In[7]:


# to get percentage
def get_perc(lst): 
    if lst[1] == 0: 
        return 0 
    else: 
        return lst[0]/lst[1]


# In[32]:


# retrieve vote information 
raw_df["percentage"] = raw_df["helpful"].apply(get_perc)
raw_df["positive_votes"] = raw_df["helpful"].apply(lambda x: x[0])
raw_df["negative_votes"] = raw_df["helpful"].apply(lambda x: x[1] - x[0])
raw_df["total_votes"] = raw_df["helpful"].apply(lambda x:x[1])


# In[33]:


raw_df.shape


# In[34]:


# will only consider all the reviews that have been voted on 
raw_df = raw_df[raw_df["total_votes"] != 0]


# In[35]:


# Convert Unix time to datetime
raw_df['datetime'] = pd.to_datetime(raw_df['unixReviewTime'], unit='s')

# Extracting date components
raw_df['year'] = raw_df['datetime'].dt.year
raw_df['month'] = raw_df['datetime'].dt.month
raw_df['day'] = raw_df['datetime'].dt.day
raw_df['weekday'] = raw_df['datetime'].dt.weekday  # Monday=0, Sunday=6


# In[28]:


raw_df.groupby("asin")["reviewerID"].count().sort_values(ascending = False).value_counts()


# In[12]:


# average of total votes 
total_votes_mean = raw_df["total_votes"].mean()

# average of positive votes (numerator)
positive_votes_mean = raw_df["positive_votes"].mean()

# average of negative votes(difference)
negative_votes_mean = raw_df["negative_votes"].mean()

print("total", total_votes_mean, "postive", positive_votes_mean, "negative", negative_votes_mean)


# In[13]:


raw_df.head()


# In[14]:


asin_focus = raw_df.groupby("asin")[["positive_votes", "negative_votes", "total_votes"]].mean().reset_index()


# In[ ]:


# px.histogram(asin_focus, x="asin", y="positive_votes")


# In[ ]:


# px.histogram(asin_focus, x="asin", y="negative_votes")


# In[15]:


raw_df["overall"].value_counts()
raw_df["overall"] = raw_df["overall"].round(0).astype(str)


# In[ ]:


# px.bar(raw_df, x = "overall", y = "positive_votes")


# In[16]:


overall_pv_df = raw_df.groupby("overall")["positive_votes"].sum().reset_index()


# In[17]:


px.bar(overall_pv_df, x="overall", y="positive_votes")


# In[18]:


overall_nv_df = raw_df.groupby("overall")["negative_votes"].sum().reset_index()
px.bar(overall_nv_df, x="overall", y="negative_votes")


# In[19]:


px.scatter(raw_df, x="percentage", y="positive_votes")


# In[ ]:


### work on actual model now 


# In[ ]:


raw_df.shape[0]


# In[ ]:


raw_df.shape


# In[20]:


# data preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Assuming raw_df is your DataFrame

# Text preprocessing
tfidf = TfidfVectorizer(max_features=5000)
vectorized_text = tfidf.fit_transform(raw_df['reviewText'])

# Convert the vectorized text to a DataFrame with the same index as raw_df
vectorized_df = pd.DataFrame(vectorized_text.toarray(), 
                             columns=tfidf.get_feature_names_out(),
                             index=raw_df.index)  # Align index

# Encode categorical features
le_reviewer = LabelEncoder()
le_asin = LabelEncoder()
raw_df['reviewerID'] = le_reviewer.fit_transform(raw_df['reviewerID'])
raw_df['asin'] = le_asin.fit_transform(raw_df['asin'])

# Combine all features
features = pd.concat([raw_df[['reviewerID', 'asin', 'overall', 'unixReviewTime']], 
                      vectorized_df], axis=1)  # Concatenate along columns

# Now, 'features' should have the same number of rows as 'raw_df'
print("Features shape:", features.shape)
print("Original DataFrame shape:", raw_df.shape)


# In[21]:


target = raw_df["percentage"]


# In[ ]:


# # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune these parameters

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[ ]:


# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Initialize the RandomForestRegressor
# model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune these parameters

# # Train the model
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')


# Assuming 'features' is your DataFrame of features and 'target' is your target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# THIS ONE IS WITHOUT TEXT AS A FEATURE 


# In[46]:


raw_df["reviewerID"].value_counts()


# In[49]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pandas as pd

# Assuming raw_df is your DataFrame

# Convert 'datetime' to numerical features
raw_df['year'] = raw_df['datetime'].dt.year
raw_df['month'] = raw_df['datetime'].dt.month
raw_df['day'] = raw_df['datetime'].dt.day
raw_df['weekday'] = raw_df['datetime'].dt.weekday

# Continue with your existing code...

# Step 1: Vectorize the Text Data
tfidf = TfidfVectorizer(max_features=5000)
vectorized_text = tfidf.fit_transform(raw_df['reviewText'])

# Step 2: Reduce Dimensionality
svd = TruncatedSVD(n_components=500)
reduced_features = svd.fit_transform(vectorized_text)

# Generate string feature names for reduced features
reduced_feature_names = [f"svd_feature_{i}" for i in range(reduced_features.shape[1])]

# Encode categorical features
le_reviewer = LabelEncoder()
le_asin = LabelEncoder()
raw_df['reviewerID'] = le_reviewer.fit_transform(raw_df['reviewerID'])
raw_df['asin'] = le_asin.fit_transform(raw_df['asin'])

# Step 3: Combine Reduced Text Features with Other Features
other_features = raw_df[['reviewerID', 'asin', 'overall', 'year', 'month']]  # Replace 'datetime' with extracted features
combined_features = pd.concat([pd.DataFrame(reduced_features, columns=reduced_feature_names), other_features.reset_index(drop=True)], axis=1)

# Ensure all column names are of type string
combined_features.columns = combined_features.columns.astype(str)

# Step 4: Split the Data
target = raw_df['percentage']  # Replace with your actual target column name
X_temp, X_test, y_temp, y_test = train_test_split(combined_features, target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

# Evaluate the model
val_mse = mean_squared_error(y_val, y_val_pred)
print(f'Validation Mean Squared Error: {val_mse}')

# Final evaluation on the test set
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f'Test Mean Squared Error: {test_mse}')


# In[ ]:


# linear regression 

including unixReviewTime
Validation Mean Squared Error: 0.1078982820100415
Test Mean Squared Error: 0.10717433729872824


# In[56]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pandas as pd

# Assuming raw_df is your DataFrame

# Convert 'overall' to a numeric type
raw_df['overall'] = pd.to_numeric(raw_df['overall'], errors='coerce')

# Step 1: Vectorize the Text Data
tfidf = TfidfVectorizer(max_features=5000)
vectorized_text = tfidf.fit_transform(raw_df['reviewText'])

# Step 2: Reduce Dimensionality
svd = TruncatedSVD(n_components=500)
reduced_features = svd.fit_transform(vectorized_text)

# Generate string feature names for reduced features
reduced_feature_names = [f"svd_feature_{i}" for i in range(reduced_features.shape[1])]

# Encode categorical features
le_reviewer = LabelEncoder()
le_asin = LabelEncoder()
raw_df['reviewerID'] = le_reviewer.fit_transform(raw_df['reviewerID'])
raw_df['asin'] = le_asin.fit_transform(raw_df['asin'])

# Step 3: Combine Reduced Text Features with Other Features
other_features = raw_df[['reviewerID', 'asin', 'overall', 'datetime']]
combined_features = pd.concat([pd.DataFrame(reduced_features, columns=reduced_feature_names), other_features.reset_index(drop=True)], axis=1)

# Ensure all column names are of type string
combined_features.columns = combined_features.columns.astype(str)

# Ensure all data types are correct
combined_features = combined_features.apply(pd.to_numeric, errors='coerce')

# Step 4: Split the Data into train, validation, and test sets
target = raw_df['percentage']  # Replace 'target_column' with your actual target column name
X_temp, X_test, y_temp, y_test = train_test_split(combined_features, target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Step 5: Train the XGBoost Regressor Model
model = XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.1, max_depth=7, random_state=42)
model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f'Validation Mean Squared Error: {val_mse}')

# Final evaluation on the test set
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f'Test Mean Squared Error: {test_mse}')


# In[ ]:


# XGBRegressor
# 0.01 with 100 was around 0.11

n_estimators: 150, learning rate: 0.1, included unixTime
Validation Mean Squared Error: 0.10678114082258353
Test Mean Squared Error: 0.10639875138479503


n_estimators: 150, learning rate: 0.1, included datetime
Validation Mean Squared Error: 0.1065981004027286
Test Mean Squared Error: 0.10623299283122345
    
n_estimator: 100, learning rate: 0.1, included datetime
Validation Mean Squared Error: 0.10700001403649959
Test Mean Squared Error: 0.10670623701065131
    
n_estimator: 300, learning rate: 0.3, included datetime
Validation Mean Squared Error: 0.11137659965017736
Test Mean Squared Error: 0.11117464390667328

n_estimator: 300, learning rate: 0.1, included datetime
Validation Mean Squared Error: 0.10657464826527628
Test Mean Squared Error: 0.10618715639932949
    
n_estimatr: 500, learning rate: 0.1, including datetime 
Validation Mean Squared Error: 0.10669503715947311
Test Mean Squared Error: 0.10649647912724372

n_estimatr: 300, learning rate: 0.1, max_depth = 5, including datetime 
Validation Mean Squared Error: 0.10647872527782656
Test Mean Squared Error: 0.10607541596432436
    
    
n_estimators=300, learning_rate=0.1, max_depth=7 including datetime 
Validation Mean Squared Error: 0.10684418000082262
Test Mean Squared Error: 0.1064496386996795


# In[55]:





# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Initialize the RandomForestRegressor
# model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune these parameters

# # Train the model
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')


# In[ ]:





# In[ ]:


len(raw_df["asin"])


# In[ ]:


features.head(-5)


# In[ ]:


raw_df.shape[0]


# In[ ]:


y_train.shape[0]


# In[ ]:





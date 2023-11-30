#!/usr/bin/env python
# coding: utf-8

# In[3]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[4]:


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


# In[5]:


f = gzip.open("/Users/linabattikha/cse158/assignment2/data/reviews_Electronics_5.json.gz")
data = [] 
for l in f: 
    data.append(json.loads(l))


# In[6]:


len(data)


# In[7]:


raw_df = pd.DataFrame(data)


# In[8]:


raw_df["helpful"].value_counts()


# In[9]:


# to get percentage
def get_perc(lst): 
    if lst[1] == 0: 
        return 0 
    else: 
        return lst[0]/lst[1]



# In[10]:


# retrieve vote information 
raw_df["percentage"] = raw_df["helpful"].apply(get_perc)
raw_df["positive_votes"] = raw_df["helpful"].apply(lambda x: x[0])
raw_df["negative_votes"] = raw_df["helpful"].apply(lambda x: x[1] - x[0])
raw_df["total_votes"] = raw_df["helpful"].apply(lambda x:x[1])


# In[11]:


raw_df.shape


# In[ ]:





# In[ ]:





# In[12]:


# will only consider all the reviews that have been voted on 
raw_df = raw_df[raw_df["total_votes"] != 0]

# keep only 50 percent of what is left 
sample_size = int(0.5 * raw_df.shape[0])

raw_df = raw_df.sample(n=sample_size, random_state=42)


# In[13]:


raw_df.shape[0]


# In[14]:


# average of total votes 
total_votes_mean = raw_df["total_votes"].mean()

# average of positive votes (numerator)
positive_votes_mean = raw_df["positive_votes"].mean()

# average of negative votes(difference)
negative_votes_mean = raw_df["negative_votes"].mean()

print("total", total_votes_mean, "postive", positive_votes_mean, "negative", negative_votes_mean)


# In[15]:


raw_df.head()


# In[16]:


asin_focus = raw_df.groupby("asin")[["positive_votes", "negative_votes", "total_votes"]].mean().reset_index()


# In[17]:


# px.histogram(asin_focus, x="asin", y="positive_votes")


# In[18]:


# px.histogram(asin_focus, x="asin", y="negative_votes")


# In[19]:


raw_df["overall"].value_counts()
raw_df["overall"] = raw_df["overall"].round(0).astype(str)


# In[20]:


# px.bar(raw_df, x = "overall", y = "positive_votes")


# In[21]:


overall_pv_df = raw_df.groupby("overall")["positive_votes"].sum().reset_index()


# In[22]:


px.bar(overall_pv_df, x="overall", y="positive_votes")


# In[23]:


overall_nv_df = raw_df.groupby("overall")["negative_votes"].sum().reset_index()
px.bar(overall_nv_df, x="overall", y="negative_votes")


# In[24]:


px.scatter(raw_df, x="percentage", y="positive_votes")


# In[25]:


### work on actual model now 


# In[26]:


raw_df.shape[0]


# In[27]:


raw_df.shape


# In[28]:


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

# Reset the index of raw_df if it has been manipulated earlier (optional, if needed)
# raw_df.reset_index(drop=True, inplace=True)

# Combine all features
features = pd.concat([raw_df[['reviewerID', 'asin', 'overall', 'unixReviewTime']], 
                      vectorized_df], axis=1)  # Concatenate along columns

# features = raw_df[['reviewerID', 'asin', 'overall', 'unixReviewTime']]


# Now, 'features' should have the same number of rows as 'raw_df'
print("Features shape:", features.shape)
print("Original DataFrame shape:", raw_df.shape)



# In[29]:


target = raw_df["percentage"]


# In[31]:


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


# In[30]:


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


# In[34]:


# THIS IS ONLY USING 50% OF THE DATA WITH ACTUAL VOTES

# Step 1: Vectorize the Text Data
tfidf = TfidfVectorizer(max_features=5000)  # Adjust the number of features
vectorized_text = tfidf.fit_transform(raw_df['reviewText'])

# Step 2: Reduce Dimensionality - THIS IS THE LATENT SEMANTIC ANALYSIS 
svd = TruncatedSVD(n_components=500)  # Adjust the number of components
reduced_features = svd.fit_transform(vectorized_text)

# Generate string feature names for reduced features
reduced_feature_names = [f"svd_feature_{i}" for i in range(reduced_features.shape[1])]

# Encode categorical features
le_reviewer = LabelEncoder()
le_asin = LabelEncoder()
raw_df['reviewerID'] = le_reviewer.fit_transform(raw_df['reviewerID'])
raw_df['asin'] = le_asin.fit_transform(raw_df['asin'])

# Step 3: Combine Reduced Text Features with Other Features
other_features = raw_df[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
combined_features = pd.concat([pd.DataFrame(reduced_features, columns=reduced_feature_names), other_features.reset_index(drop=True)], axis=1)

# Ensure all column names are of type string
combined_features.columns = combined_features.columns.astype(str)

# Step 4: Split the Data
target = raw_df['percentage']  # Replace 'target_column' with your actual target column name
X_train, X_test, y_train, y_test = train_test_split(combined_features, target, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[38]:


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
tfidf = TfidfVectorizer(max_features=5000)  # Adjust the number of features
vectorized_text = tfidf.fit_transform(raw_df['reviewText'])

# Step 2: Reduce Dimensionality
svd = TruncatedSVD(n_components=500)  # Adjust the number of components
reduced_features = svd.fit_transform(vectorized_text)

# Generate string feature names for reduced features
reduced_feature_names = [f"svd_feature_{i}" for i in range(reduced_features.shape[1])]

# Encode categorical features
le_reviewer = LabelEncoder()
le_asin = LabelEncoder()
raw_df['reviewerID'] = le_reviewer.fit_transform(raw_df['reviewerID'])
raw_df['asin'] = le_asin.fit_transform(raw_df['asin'])

# Step 3: Combine Reduced Text Features with Other Features
other_features = raw_df[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
combined_features = pd.concat([pd.DataFrame(reduced_features, columns=reduced_feature_names), other_features.reset_index(drop=True)], axis=1)

# Ensure all column names are of type string
combined_features.columns = combined_features.columns.astype(str)

# Ensure all data types are correct
combined_features = combined_features.apply(pd.to_numeric, errors='coerce')

# Step 4: Split the Data
target = raw_df['percentage']  # Replace 'target_column' with your actual target column name
X_train, X_test, y_train, y_test = train_test_split(combined_features, target, test_size=0.2, random_state=42)

# Step 5: Train the XGBoost Regressor Model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


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





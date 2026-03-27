#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

client_df = pd.read_csv("client_data.csv")
client_df.head()


# In[2]:


client_df.columns


# In[3]:


client_df['churn'].value_counts()


# In[4]:


client_df['churn'].value_counts(normalize=True)


# In[5]:


client_df.info()


# In[6]:


date_cols = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal']

for col in date_cols:
    client_df[col] = pd.to_datetime(client_df[col], errors='coerce')


# In[7]:


client_df.describe()


# In[8]:


client_df.describe(include='object')


# In[9]:


client_df.describe(include=['object', 'string'])


# In[10]:


6754 / 14606


# In[11]:


11955 / 14606


# In[12]:


7097 / 14606


# In[13]:


client_df['churn'].value_counts()
client_df['churn'].value_counts(normalize=True)


# Churn Distribution Analysis
# 
# The churn variable is highly imbalanced. Approximately 90.3% of customers did not churn, while only 9.7% of customers churned.
# 
# This indicates that churn is a relatively rare event within the dataset. From a business perspective, although the proportion of churned customers is small, the financial impact may still be significant depending on margin and consumption levels.
# 
# From a modeling perspective, the class imbalance may require special handling techniques such as resampling, class weighting, or evaluation metrics beyond accuracy.

# In[14]:


client_df.groupby('churn').mean(numeric_only=True)


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='churn', y='num_years_antig', data=client_df)
plt.title("Tenure vs Churn")
plt.show()


# In[17]:


sns.boxplot(x='churn', y='cons_12m', data=client_df)
plt.title("12 Month Consumption vs Churn")
plt.show()


# In[18]:


sns.boxplot(x='churn', y='net_margin', data=client_df)
plt.title("Net Margin vs Churn")
plt.show()


# In[19]:


client_df.groupby('churn').mean(numeric_only=True)


# Tenure and Contract Characteristics
# 
# Churned customers have a lower average tenure (4.63 years) compared to non-churned customers (5.04 years), indicating that newer customers are more likely to churn. This suggests that early-stage retention strategies could be particularly effective.
# 
# Additionally, churned customers exhibit slightly higher contracted maximum power levels. This may imply greater exposure to fixed capacity charges, potentially contributing to price sensitivity and switching behavior.
# 
# Combined with earlier findings on lower consumption and higher discounts, the results suggest that churn is influenced by pricing structure and customer lifecycle stage.

# In[20]:


pd.crosstab(client_df['channel_sales'], client_df['churn'], normalize='index')


# In[21]:


pd.crosstab(client_df['origin_up'], client_df['churn'], normalize='index')


# Churn at PowerCo appears driven by a combination of lifecycle stage, consumption behavior, pricing structure, and acquisition strategy. Customers acquired through specific channels and origins exhibit materially higher churn rates. Additionally, lower consumption and shorter tenure significantly correlate with churn probability. Strategic adjustments in acquisition mix, pricing structure, and early-stage retention interventions may meaningfully reduce churn.

# In[22]:


df.info()
df.dtypes


# In[23]:


client_df.info()
client_df.dtypes


# In[24]:


price_df.info()
price_df.dtypes


# In[25]:


import pandas as pd
import numpy as np


# In[26]:


client_df = pd.read_csv("client_data.csv")
price_df = pd.read_csv("price_data.csv")
churn_df = pd.read_csv("churn_data.csv")


# In[27]:


import os
os.listdir()


# In[28]:


churn


# In[29]:


client_df.columns


# In[30]:


client_df = pd.read_csv("client_data.csv")
price_df = pd.read_csv("price_data.csv")


# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

client_df = pd.read_csv("client_data.csv")
price_df = pd.read_csv("price_data.csv")


# In[32]:


client_df.info()
client_df.dtypes


# In[33]:


client_df['date_activ'] = pd.to_datetime(client_df['date_activ'])
client_df['date_end'] = pd.to_datetime(client_df['date_end'])


# In[34]:


price_df.info()
price_df.dtypes


# In[35]:


price_df['price_date'] = pd.to_datetime(price_df['price_date'])


# In[36]:


client_df.describe()
client_df.nunique()
client_df.isnull().sum()


# Missing Values Analysis
# 
# We checked for missing values using .isnull().sum() and found that all columns contain zero missing values.
# 
# This indicates strong data completeness and no immediate need for imputation or data cleaning.

# In[37]:


price_df.describe()
price_df.nunique()
price_df.isnull().sum()


# In[38]:


price_df.info()


# In[39]:


price_df['price_date'] = pd.to_datetime(price_df['price_date'])


# In[40]:


sns.histplot(price_df['price_off_peak_var'], bins=50)
plt.title("Off Peak Variable Price Distribution")
plt.show()

sns.histplot(price_df['price_peak_var'], bins=50)
plt.title("Peak Variable Price Distribution")
plt.show()

sns.histplot(price_df['price_off_peak_fix'], bins=50)
plt.title("Off Peak Fixed Price Distribution")
plt.show()


# In[41]:


price_df.groupby('price_date')['price_off_peak_var'].mean().plot()
plt.title("Average Off Peak Variable Price Over Time")
plt.show()


# Exploratory Data Analysis Summary
# 
# Both datasets contain complete data with no missing values.
# 
# Client dataset includes a mix of categorical and numerical variables.
# 
# Churn is imbalanced (~90% non-churn).
# 
# Consumption-related variables appear right-skewed.
# 
# Pricing variables show variation across peak and off-peak periods.
# 
# No major structural data issues were identified.

# In[42]:


client_df = client_df.drop(columns=['id'])


# In[43]:


price_df = price_df.drop(columns=['id'])


# In[44]:


client_df.nunique()
price_df.nunique()


# In[45]:


date_cols = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal']

for col in date_cols:
    client_df[col] = pd.to_datetime(client_df[col])


# In[46]:


client_df['activation_year'] = client_df['date_activ'].dt.year
client_df['activation_month'] = client_df['date_activ'].dt.month

client_df['renewal_year'] = client_df['date_renewal'].dt.year
client_df['renewal_month'] = client_df['date_renewal'].dt.month


# In[47]:


client_df['tenure_days'] = (client_df['date_end'] - client_df['date_activ']).dt.days
client_df['tenure_years'] = client_df['tenure_days'] / 365


# In[48]:


client_df['total_consumption'] = client_df['cons_12m'] + client_df['cons_gas_12m']


# In[49]:


client_df['consumption_diff'] = client_df['forecast_cons_12m'] - client_df['cons_12m']


# In[50]:


price_df['var_price_diff'] = price_df['price_peak_var'] - price_df['price_off_peak_var']
price_df['fix_price_diff'] = price_df['price_peak_fix'] - price_df['price_off_peak_fix']


# In[51]:


price_agg = price_df.groupby('id').mean().reset_index()


# In[52]:


final_df = client_df.merge(price_agg, on='id', how='left')


# In[53]:


price_df.columns


# In[54]:


client_df['price_diff_energy'] = (
    client_df['forecast_price_energy_peak'] - 
    client_df['forecast_price_energy_off_peak']
)


# In[55]:


client_df['total_forecast_price'] = (
    client_df['forecast_price_energy_off_peak'] +
    client_df['forecast_price_energy_peak'] +
    client_df['forecast_price_pow_off_peak']
)


# In[56]:


client_df['effective_price_after_discount'] = (
    client_df['total_forecast_price'] *
    (1 - client_df['forecast_discount_energy'])
)


# In[57]:


client_df['total_consumption'] = (
    client_df['cons_12m'] +
    client_df['cons_gas_12m']
)


# In[58]:


client_df['consumption_gap'] = (
    client_df['forecast_cons_12m'] -
    client_df['cons_12m']
)


# In[59]:


client_df['date_activ'] = pd.to_datetime(client_df['date_activ'])
client_df['date_end'] = pd.to_datetime(client_df['date_end'])

client_df['tenure_days'] = (
    client_df['date_end'] -
    client_df['date_activ']
).dt.days


# In[60]:


client_df['is_multi_product'] = (
    client_df['nb_prod_act'] > 1
).astype(int)


# In[61]:


client_df = client_df.drop(columns=[
    'date_activ',
    'date_end',
    'date_modif_prod',
    'date_renewal'
])


# In[62]:


client_df = client_df.drop(columns=['id'])


# In[63]:


client_df.columns


# In[64]:


client_df = client_df.drop(columns=['id'], errors='ignore')


# In[65]:


client_df.nunique().sort_values()


# In[66]:


date_cols = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal']

for col in date_cols:
    client_df[col] = pd.to_datetime(client_df[col])


# In[67]:


client_df.columns.tolist()


# In[68]:


date_cols = [col for col in client_df.columns if 'date' in col]

print(date_cols)


# In[69]:


for col in date_cols:
    client_df[col] = pd.to_datetime(client_df[col], errors='coerce')


# In[70]:


client_df.head()


# In[71]:


client_df.columns


# In[72]:


client_df.nunique().sort_values()


# In[73]:


client_df = pd.get_dummies(
    client_df,
    columns=['channel_sales', 'origin_up'],
    drop_first=True
)


# In[74]:


client_df['has_gas'] = client_df['has_gas'].astype(int)


# In[75]:


client_df.nunique().sort_values()


# In[76]:


client_df = client_df.drop(columns=['column_name'], errors='ignore')


# In[77]:


client_df['total_consumption'] = client_df['cons_12m'] + client_df['cons_gas_12m']
client_df['consumption_gap'] = client_df['forecast_cons_12m'] - client_df['cons_12m']


# In[78]:


client_df['price_diff_energy'] = (
    client_df['forecast_price_energy_peak'] -
    client_df['forecast_price_energy_off_peak']
)

client_df['total_forecast_price'] = (
    client_df['forecast_price_energy_peak'] +
    client_df['forecast_price_energy_off_peak']
)

client_df['effective_price_after_discount'] = (
    client_df['total_forecast_price'] *
    (1 - client_df['forecast_discount_energy'])
)


# In[79]:


client_df['is_multi_product'] = (client_df['nb_prod_act'] > 1).astype(int)

client_df['margin_ratio'] = (
    client_df['margin_net_pow_ele'] /
    client_df['margin_gross_pow_ele']
)


# In[80]:


import numpy as np

client_df['log_total_consumption'] = np.log1p(client_df['total_consumption'])
client_df['log_tenure'] = np.log1p(client_df['tenure_days'])


# In[81]:


client_df = pd.get_dummies(
    client_df,
    columns=['channel_sales', 'origin_up'],
    drop_first=True
)

client_df['has_gas'] = client_df['has_gas'].astype(int)


# In[82]:


final_df = pd.merge(
    client_df,
    price_df,
    on='id',
    how='left'
)


# In[83]:


client_df.columns.tolist()


# In[84]:


# Find categorical columns automatically
cat_cols = client_df.select_dtypes(include=['object']).columns

print(cat_cols)


# In[85]:


client_df = pd.get_dummies(client_df, columns=cat_cols, drop_first=True)


# In[86]:


client_df.info()


# In[87]:


client_df['margin_ratio'] = client_df['margin_ratio'].fillna(0)


# In[88]:


client_df.isnull().sum()


# In[89]:


bool_cols = client_df.select_dtypes(include=['bool']).columns

client_df[bool_cols] = client_df[bool_cols].astype(int)


# In[90]:


client_df.info()
client_df.isnull().sum().sum()


# Final Feature Engineering Summary
# 
# The dataset was prepared for churn prediction using a structured feature engineering approach:
# 
# Removed non-predictive columns such as customer identifiers and raw date variables.
# 
# Extracted time-based features including tenure (days and years), activation year/month, and renewal year/month.
# 
# Created new consumption features such as total consumption and consumption gap.
# 
# Engineered price sensitivity features including energy price difference and effective price after discount.
# 
# Added customer value indicators such as multi-product flag and margin ratio.
# 
# Applied log transformations to skewed variables to improve model stability.
# 
# Encoded categorical variables using one-hot encoding.
# 
# Handled missing values and ensured all features are numeric and model-ready.
# 
# The resulting dataset is fully cleaned, enriched, and ready for predictive modeling to identify churn drivers.

# In[91]:


client_df.shape


# In[ ]:





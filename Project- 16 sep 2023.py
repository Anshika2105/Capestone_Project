#!/usr/bin/env python
# coding: utf-8

# # OLIST Data Science Adoption Strategy
# 

# **The Olist dataset ?**
# 
# This is a Brazilian ecommerce public dataset of orders made at Olist Store. The dataset has information of **100k orders from 2016 to 2018** made at multiple marketplaces in Brazil. 
# Its features allows viewing an order from multiple dimensions: from order status, price, payment and freight performance to  product attributes and finally reviews written by customers.
# 
# ![](https://storage.googleapis.com/kaggle-datasets-images/55151/105464/d59245a7014a35a35cc7f7b721de4dae/dataset-cover.png?t=2018-09-21-16-21-21)
# 
# Link to dataset - https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

# ### Schema

# ![](https://i.imgur.com/HRhd2Y0.png)

# ### Outline of Project

# 1. Downloading dataset and uploading CSV files to Jupyter Notebook.
# 2. Data Preprocessing - Reading, Merging and Cleaning
# 3. Discussing the Business Metrics - Analysis and Visualization
# 4. Products - Analysis and Visualization
# 5. Customers - Analysis and Visualization
# 6. Sellers - Analysis and Visualization
# 7. Reviews Sentiment - Analysis and Visualization
# 

# In[1]:


#import the required libraries
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")


# # Data Preprocessing

# The dataset contains 6 separate CSV files containing data of customers, orders, order items, reviews and products. Taking the the files one at a time, data preprocessing is done below :-

# ## 1.1 Reading Orders Data

# The orders data is in two separate files so reading each separately

# In[2]:


orders_data = pd.read_csv('orders_dataset.csv')

orders_data.head(5)


# In[3]:


orders_data.shape


# In[4]:


orders_data.dtypes


# In[5]:


orders_data.isnull().sum()


# In[6]:


orders_items_data = pd.read_csv('order_items_dataset.csv')

orders_items_data.head(5)


# In[7]:


orders_items_data.shape


#  Combining the the two order data sets : orders_data and orders_items_data to get a mapping of customer to the products purchased and the price of the products.

# In[8]:


cols= [
    'order_id',
    'customer_id',
    'order_purchase_timestamp',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
    'order_item_id',
    'product_id',
    'price',
    'freight_value'
]

complete_order = pd.merge(
    left= orders_items_data,
    right= orders_data,
    on= 'order_id',
    how= 'inner'
)[cols]

complete_order.head(100)


# In[9]:


complete_order.shape


# Changing Date Time Formats

# In[10]:


complete_order['order_purchase_timestamp'] = pd.to_datetime(complete_order['order_purchase_timestamp'])
complete_order['order_delivered_customer_date'] = pd.to_datetime(complete_order['order_delivered_customer_date'])
complete_order['order_estimated_delivery_date'] = pd.to_datetime(complete_order['order_estimated_delivery_date'])


# In[11]:


complete_order.dtypes


# ### 1.1.1 Filtering for dates between Jan 2017 - Sep 2018 
# (since quality is good and mentioned in problem statement)

# In[12]:


import datetime 
complete_order_indate= complete_order.loc[(complete_order['order_purchase_timestamp'] >= '2017-01-01')
                                       & (complete_order['order_purchase_timestamp'] < '2018-10-01') ]


# In[13]:


complete_order_indate


# Thus, 370 entries are removed from the dataframe after filtering

# ### 1.1.1 Dealing with missing values

# In[14]:


complete_order_indate.isnull().sum()


# FILLING THE ORDER DELIVERED EMPTY DATE WITH ESTIMATED DELIVERY DATE AS IT WAS GIVEN IN THE PROBLEM STATEMENT THAT ALL ITEMS ARE ASSUMED TO BE DELIVERED

# In[15]:


complete_order_indate["order_delivered_customer_date"].fillna(complete_order_indate["order_estimated_delivery_date"], inplace=True)


# In[16]:


complete_order_indate.isnull().sum()


# In[17]:


complete_order_indate.duplicated().sum()


# ## 1.2 Reading Products Data

# In[18]:


products_data = pd.read_csv('products_dataset.csv')

products_data.head(5)


# ### 1.2.1 Merging Products Data with Filtered Orders Data

# This will tell us about details of product Categories

# In[19]:


cols2= [
    'order_id',
    'customer_id',
    'order_purchase_timestamp',
    'order_delivered_customer_date',
    'order_item_id',
    'product_id',
    'price',
    'freight_value',
    'product_category_name'
]

product_filtered = pd.merge(
    left= complete_order_indate,
    right= products_data,
    on= 'product_id',
    how= 'inner'
)[cols2]

product_filtered.head(100)


# In[20]:


product_filtered.isna().sum()


# ### Dealing with Missing Values 

# Filling the missing product categories with new category named Unknown

# In[21]:


product_filtered["product_category_name"]=product_filtered["product_category_name"].fillna("Unknown")


# In[22]:


product_filtered.isna().sum()


# In[23]:


product_filtered.shape


# In[24]:


product_filtered.sample(10)


# In[25]:


product_filtered.duplicated().sum()


# ## 1.3 Reading Customer Data

# In[26]:


customer_data = pd.read_csv("customers_dataset.csv")

# Look at the top 5 records of data

customer_data.head()


# In[27]:


print("Number of unique customers:- ",len(customer_data.customer_unique_id.unique()))


# ### 1.3.1 Merging Customer data with Product Filtered

# In[28]:


cols3= [
    'order_id',
    'customer_id',
    'order_purchase_timestamp',
    'order_delivered_customer_date',
    'order_item_id',
    'product_id',
    'price',
    'freight_value',
    'product_category_name',
    'customer_city',
    'customer_state'
]

customer_filtered = pd.merge(
    left= product_filtered,
    right= customer_data,
    on= 'customer_id',
    how= 'inner'
)[cols3]

customer_filtered.head(100)


# In[29]:


customer_filtered.isnull().sum()


# In[30]:


customer_filtered.duplicated().sum()


# ## 1.4 Reading Reviews Data

# What is reviews data??

# In[31]:


reviews_data = pd.read_csv('order_reviews_dataset.csv')

reviews_data.head(5)


# In[32]:


reviews_data.shape


# ### 1.4.1 Merging Reviews with Customer Filtered

# In[33]:


cols4= [
    'order_id',
    'customer_id',
    'order_purchase_timestamp',
    'order_delivered_customer_date',
    'order_item_id',
    'product_id',
    'price',
    'freight_value',
    'product_category_name',
    'customer_city',
    'customer_state',
    'review_score'
]

reviews_filtered = pd.merge(
    left= customer_filtered,
    right= reviews_data,
    on= 'order_id',
    how= 'inner'
)[cols4]

reviews_filtered.head(100)


# In[34]:


reviews_filtered.shape


# In[35]:


reviews_filtered.sample(10)


# In[36]:


reviews_filtered.isnull().sum()


# In[37]:


reviews_filtered.duplicated().sum()


# Droping duplicate rows from reviews filtered dataframe

# In[38]:


reviews_filtered.drop_duplicates(inplace=True)


# In[39]:


reviews_filtered.duplicated().sum()


# ## 1.5 Seller Data

# In[40]:


seller_data = pd.read_csv('sellers_dataset.csv')

seller_data.head(5)


# In[41]:


seller_data.shape


# In[42]:


len((seller_data.seller_id).unique())


# In[43]:


seller_data.dtypes


# In[44]:


seller_data.isnull().sum()


# The Sellers data has no common column to merge with filtered datasets, so it will be dealt separately

# In[45]:


seller_data.duplicated().sum()


# ## 1.6 Merging it all Together 
# (if we ignore the date filter to observe how tohe overall data looks like)

# In[46]:


df_item = pd.read_csv("order_items_dataset.csv")
df_reviews = pd.read_csv("order_reviews_dataset.csv")
df_orders = pd.read_csv("orders_dataset.csv")
df_products = pd.read_csv("products_dataset.csv")
df_geolocation = pd.read_csv("geolocation_dataset.csv")
df_sellers = pd.read_csv("sellers_dataset.csv")
df_order_pay = pd.read_csv("order_payments_dataset.csv")
df_customers = pd.read_csv("customers_dataset.csv")
df_category = pd.read_csv("Product_category_name_translation.csv")


# In[47]:


df_train = df_orders.merge(df_item, on='order_id', how='left')
df_train = df_train.merge(df_order_pay, on='order_id', how='outer', validate='m:m')
df_train = df_train.merge(df_reviews, on='order_id', how='outer')
df_train = df_train.merge(df_products, on='product_id', how='outer')
df_train = df_train.merge(df_customers, on='customer_id', how='outer')
df_train = df_train.merge(df_sellers, on='seller_id', how='outer')

print(df_train.shape)


# ### Understanding the Data - Data Types, Values, Entropy

# In[48]:


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

def cross_heatmap(df, cols, normalize=False, values=None, aggfunc=None):
    temp = cols
    cm = sns.light_palette("green", as_cmap=True)
    return pd.crosstab(df[temp[0]], df[temp[1]], 
                       normalize=normalize, values=values, aggfunc=aggfunc).style.background_gradient(cmap = cm)


# In[49]:


resumetable(df_train)


# *The entropy value provided for each column in the table represents the amount of variability or unpredictability in the values of that respective column. In the context of data analysis, entropy is a measure of the information content or the degree of uncertainty associated with the values in a particular column.*
# 
# *A higher entropy value indicates greater variability or uncertainty, meaning that the values in that column are more diverse and less predictable. Conversely, a lower entropy value suggests less variability or uncertainty, implying that the values in the column are more consistent and predictable.*

# # Analysis and Visualization

# ## 2.1 What are the important e-commerce Metrics & KPIs

# A metric is any quantifiable, consistently defined measurement of website performance. KPI stands for key performance indicator. While all metrics have their value, a KPI is especially important to keep track of as these are the numbers you track for growth.
# Not all ecommerce metrics are as valuable as others but identifying the key performance indicators (KPIs)  will helps in improving online store's performance.
# 
# 

# In[50]:


col = ['customer_unique_id', 'price', 'order_item_id', 'order_purchase_timestamp']
orders = df_train[col]


# In[51]:


orders


# In[52]:


#converting the type of Invoice Date Field from string to datetime.
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

#creating YearMonth field for the ease of reporting and visualization
orders['InvoiceYearMonth'] = orders['order_purchase_timestamp'].map(lambda date: 100*date.year + date.month)

#calculate Revenue for each row and create a new dataframe with YearMonth - Revenue columns
orders['Revenue'] = orders['price'] * orders['order_item_id']
orders_revenue = orders.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()
orders.head()


# ### Monthly Recurring Revenue

# Monthly Recurring Revenue (MRR) is the income that a company expects to receive in payments on a monthly basis. MRR is a critical revenue metric that helps subscription companies to understand their overall business health profitability by keeping a close eye on monthly cash flow.

# In[53]:


#X and Y axis inputs for Plotly graph. We use Scatter for line graphs
plot_data = [
    go.Scatter(
        x=orders_revenue['InvoiceYearMonth'],
        y=orders_revenue['Revenue'],
        mode = 'lines+markers'
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Revenue', width=900, height=500
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


# *INSIGHT - The monthly revenue seems to consistently increased from Dec 2016 to Nov 2017 and then became mnearly stangnated till Aug 2018*

# ### Monthly Growth Rate

# Month-over-Month (MoM) is the smallest unit of measurement used to objectively capture the rate of growth in a business. This metric scales up to Quarter on Quarter and Year on Year growth tracking to give you an idea of rates of growth over varied time scales. It’s most commonly used for projections by early-stage companies,

# In[54]:


#using pct_change() function to see monthly percentage change
orders_revenue['MonthlyGrowth'] = orders_revenue['Revenue'].pct_change()

#showing first 5 rows
orders_revenue.head(15)


# In[55]:


#visualization - line graph
plot_data = [
    go.Scatter(
        x=orders_revenue.query("InvoiceYearMonth > 201701")['InvoiceYearMonth'],
        y=orders_revenue.query("InvoiceYearMonth > 201701")['MonthlyGrowth'],
        mode = 'lines+markers'
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Growth Rate', width=900, height=500
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


# *INSIGHT - In the initial months of 2017, growth rate remained above 40% till May with the exception of April. A root cause analysis is needed to understand what happened in April which caused the dip. Again there was a similar dip in growth rate in June 2017 post which the growth rate remained around 15-20% till September. There was sudden increase in November followed by a sudden dip in December post which the growth rate remained around 0 percent*

# ### Monthly Active Customers

# Monthly active users (also known as MAU) is a marketing metric used to determine the number of unique users that visit a website or use an app within a month. 
# 
# Monitoring Monthly Active Users (MAU) is important as it provides insights into the number of users who actively engage with a product or service over a period of time and allowing businesses to measure and improve customer engagement and retention.
# 
# MAU can serve as a key performance indicator (KPI) for businesses, helping them measure the success of their marketing efforts and product development strategies. By analyzing changes in MAUs over time, companies can identify trends, understand the impact of new features or marketing campaigns, and make data-driven decisions to optimize their products and services.
# 
# *Here we are using Monthly Actice Customers as we have the data of customers that are transacting ( as compared to users coming on the platform*

# In[56]:


#creating a new dataframe with UK customers only
#tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)

#creating monthly active customers dataframe by counting unique Customer IDs
orders_monthly_active = orders.groupby('InvoiceYearMonth')['customer_unique_id'].nunique().reset_index()

#print the dataframe
orders_monthly_active.head()


# In[57]:


#plotting the output
plot_data = [
    go.Bar(
        x=orders_monthly_active['InvoiceYearMonth'],
        y=orders_monthly_active['customer_unique_id'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Active Customers',width=900, height=500
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


# *INSIGHT - The monthly active customers seems to consistently increased from Dec 2016 to Nov 2017 and then became nearly stangnated till Aug 2018*

# ### Monthly Orders  

# Again here we are measuring the total number of orders inb each month. If we could have the data of cancelled orders, we can calcultae fulfillment rate which is one of key metrics to check the quality of orders placed, pricing issues, delivery issues etc. related to order
# 
# *Note: For the sake of analysis, we have considered all the orders to be delivered*

# In[58]:


#create a new dataframe for no. of order by using quantity field
orders_monthly_sales = orders.groupby('InvoiceYearMonth')['order_item_id'].sum().reset_index()

#print the dataframe
orders_monthly_sales.head()


# In[59]:


#plot
plot_data = [
    go.Bar(
        x=orders_monthly_sales['InvoiceYearMonth'],
        y=orders_monthly_sales['order_item_id'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Total # of Order',width=900, height=500
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


# *INSIGHT - The total number of orders in each month seems to consistently increased from Dec 2016 to Nov 2017 and then became nearly stangnated till Aug 2018*

# ### Monthly Average Orders Values

# In[60]:


# create a new dataframe for average revenue by taking the mean of it
orders_monthly_order_avg = orders.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()

#print the dataframe
orders_monthly_order_avg.head()


# In[61]:


#plot the bar chart
plot_data = [
    go.Bar(
        x=orders_monthly_order_avg['InvoiceYearMonth'],
        y=orders_monthly_order_avg['Revenue'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Order Average Values',width=900, height=500
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()
     


# *INSIGHT : Except for the month of December 2016, the average order values has remained more or less stagnated around 130. This means that the buying behaviour of customer has remained the same and costly items on the platform did not have a significant impact on the sale and average order value*

# ### New vs Repeat Customers

# A repeat customer is someone who has purchased twice or more from your site. Usually, there’s no time limit on what counts as a repeat customer. If someone made their first purchase with your site one year ago and only made their second purchase last week, they still count as a repeat customer.
# 
# However many companies use some timeframe like 8-12 months beyond which if a cutomer repeats order, it may be considered as new 

# In[62]:


#create a dataframe contaning CustomerID and first purchase date
orders_min_purchase = orders.groupby('customer_unique_id').order_purchase_timestamp.min().reset_index()
orders_min_purchase.columns = ['customer_unique_id','MinPurchaseDate']
orders_min_purchase['MinPurchaseYearMonth'] = orders_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)

#merge first purchase date column to our main dataframe (tx_uk)
orders = pd.merge(orders, orders_min_purchase, on='customer_unique_id')


# In[63]:


#create a column called User Type and assign Existing 
#if User's First Purchase Year Month before the selected Invoice Year Month
orders['UserType'] = 'New'
orders.loc[orders['InvoiceYearMonth']> orders['MinPurchaseYearMonth'],'UserType'] = 'Existing'

#calculate the Revenue per month for each user type
orders_user_type_revenue = orders.groupby(['InvoiceYearMonth','UserType'])['Revenue'].sum().reset_index()
orders.head()


# In[64]:


orders.UserType.value_counts()


# In[65]:


#filtering the dates and plot the result
orders_user_type_revenue = orders_user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")
plot_data = [
    go.Scatter(
        x=orders_user_type_revenue.query("UserType == 'Existing'")['InvoiceYearMonth'],
        y=orders_user_type_revenue.query("UserType == 'Existing'")['Revenue'],
        name = 'Existing'
    ),
    go.Scatter(
        x=orders_user_type_revenue.query("UserType == 'New'")['InvoiceYearMonth'],
        y=orders_user_type_revenue.query("UserType == 'New'")['Revenue'],
        name = 'New'
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New vs Existing Users'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()
     


# *INSIGHT: New customers constantly grew from Jan 2017 to November 2017 leading to increase in the number of monthly orders as well. However, the growth rate of repeat customer is close to 0.01 and overall repeat customer transactions are also low leading to high customer churn. This is a very important signal to the business as it means that customers are not willing to come back to the platform again and this will lead to loss of business in the long run* 

# ### New Customer Ratio

# In[66]:


#create a dataframe that shows new user ratio - we also need to drop NA values (first month new user ratio is 0)
orders_user_ratio = orders.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['customer_unique_id'].nunique()/orders.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['customer_unique_id'].nunique() 
orders_user_ratio = orders_user_ratio.reset_index()
orders_user_ratio = orders_user_ratio.dropna()

#print the dafaframe
orders_user_ratio

#plot the result

plot_data = [
    go.Bar(
        x=orders_user_ratio.query("InvoiceYearMonth>201701 and InvoiceYearMonth<201806")['InvoiceYearMonth'],
        y=orders_user_ratio.query("InvoiceYearMonth>201701 and InvoiceYearMonth<201806")['customer_unique_id'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New Customer Ratio'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


# *INSIGHT : Except of the initial months of 2017, new customer ratio has slowly decreased from above 100 to 50. Which means that new cusrtomers are 100-50 times the repeat customers*

# ### Monthly Retention Rate

# Customer retention rate is the percentage of existing customers that continue buying from your brand over a given period of time. 

# In[67]:


#identify which users are active by looking at their revenue per month
orders_user_purchase = orders.groupby(['customer_unique_id','InvoiceYearMonth'])['Revenue'].sum().reset_index()

#create retention matrix with crosstab
orders_retention = pd.crosstab(orders_user_purchase['customer_unique_id'], orders_user_purchase['InvoiceYearMonth']).reset_index()

orders_retention.head()


# In[68]:


#create an array of dictionary which keeps Retained & Total User count for each month
months = orders_retention.columns[2:]
retention_array = []
for i in range(len(months)-1):
    retention_data = {}
    selected_month = months[i+1]
    prev_month = months[i]
    retention_data['InvoiceYearMonth'] = int(selected_month)
    retention_data['TotalUserCount'] = orders_retention[selected_month].sum()
    retention_data['RetainedUserCount'] = orders_retention[(orders_retention[selected_month]>0) & (orders_retention[prev_month]>0)][selected_month].sum()
    retention_array.append(retention_data)
    
#convert the array to dataframe and calculate Retention Rate
orders_retention = pd.DataFrame(retention_array)
orders_retention['RetentionRate'] = orders_retention['RetainedUserCount']/orders_retention['TotalUserCount']
     


# In[69]:


#plot the retention rate graph
plot_data = [
    go.Scatter(
        x=orders_retention.query("InvoiceYearMonth<201806")['InvoiceYearMonth'],
        y=orders_retention.query("InvoiceYearMonth<201806")['RetentionRate'],
        name="organic"
    )
    
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Retention Rate'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


# *INSIGHT: As discussed earlier, there is vbery high customer churn and hence low overall customer retention remaining less than 0.008 throughout the timeframe*

# ### Retention Cohort

# Cohort retention rate is a metric that measures the percentage of customers who continue to do business with a company over time. To calculate cohort retention rate, a business first defines a cohort of customers based on a common characteristic or behavior, such as the month in which they made their first purchase.

# In[70]:


#create our retention table again with crosstab() and add firs purchase year month view
orders_retention = pd.crosstab(orders_user_purchase['customer_unique_id'], orders_user_purchase['InvoiceYearMonth']).reset_index()
orders_retention = pd.merge(orders_retention,orders_min_purchase[['customer_unique_id','MinPurchaseYearMonth']],on='customer_unique_id')
new_column_names = [ 'm_' + str(column) for column in orders_retention.columns[:-1]]
new_column_names.append('MinPurchaseYearMonth')
orders_retention.columns = new_column_names

#create the array of Retained users for each cohort monthly
retention_array = []
for i in range(len(months)):
    retention_data = {}
    selected_month = months[i]
    prev_months = months[:i]
    next_months = months[i+1:]
    for prev_month in prev_months:
        retention_data[prev_month] = np.nan
        
    total_user_count = orders_retention[orders_retention.MinPurchaseYearMonth ==  selected_month].MinPurchaseYearMonth.count()
    retention_data['TotalUserCount'] = total_user_count
    retention_data[selected_month] = 1 
    
    query = "MinPurchaseYearMonth == {}".format(selected_month)
    

    for next_month in next_months:
        new_query = query + " and {} > 0".format(str('m_' + str(next_month)))
        retention_data[next_month] = np.round(orders_retention.query(new_query)['m_' + str(next_month)].sum()/total_user_count,2)
    retention_array.append(retention_data)
    
orders_retention = pd.DataFrame(retention_array)
orders_retention.index = months

#showing new cohort based retention table
orders_retention.head()


# ### How is the company performing in terms of Delivery ? 
# #### Analysing Delivery Times 

# In[71]:


# Variation from Estimated Delivery Time

df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])
df_orders['order_approved_at'] = pd.to_datetime(df_orders['order_approved_at'])
df_orders['order_estimated_delivery_date'] = pd.to_datetime(df_orders['order_estimated_delivery_date'])
df_orders['order_delivered_customer_date'] = pd.to_datetime(df_orders['order_delivered_customer_date'])
# Calculate differences in hours
df_orders['delivery_time'] = (df_orders['order_delivered_customer_date'] - df_orders['order_approved_at']).dt.total_seconds() / 86400
df_orders['estimated_delivery_time'] = (df_orders['order_estimated_delivery_date'] - df_orders['order_approved_at']).dt.total_seconds() / 86400
# Delivery estimated time and actual delivery time
plt.figure(figsize=(10,4))
plt.title("Delivery time in days")
plt.xlim(-10, 200)

ax1 = sns.kdeplot(df_orders['delivery_time'].dropna(), color="#D84E30", label='Delivery time')
ax2 = sns.kdeplot(df_orders['estimated_delivery_time'].dropna(), color="#7E7277", label='Estimated delivery time')


# *INSIGHT : There is significant difference between the median expected delivery time and the actual delivery time. This could be one of the major cause of poor customer experience leading to high churn*

# ![](https://www.researchgate.net/profile/Camila-Lorenz/publication/341914696/figure/fig1/AS:898814760198144@1591305546627/Map-of-Brazil-showing-the-states-belonging-to-each-region-Acronyms-for-each-state.jpg)

# In[72]:


#The objective here is to get the mean value when Seller is from X State and Customer from Y State.
# I will select only the more frequent states to a better view 

# Seting regions
sudeste = ['SP', 'RJ', 'ES','MG']
nordeste= ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA']
norte =  ['AM', 'RR', 'AP', 'PA', 'TO', 'RO', 'AC']
centro_oeste = ['MT', 'GO', 'MS' ,'DF' ]
sul = ['SC', 'RS', 'PR']

df_train.loc[df_train['customer_state'].isin(sudeste), 'cust_Region'] = 'Southeast'
df_train.loc[df_train['customer_state'].isin(nordeste), 'cust_Region'] = 'Northeast'
df_train.loc[df_train['customer_state'].isin(norte), 'cust_Region'] = 'North'
df_train.loc[df_train['customer_state'].isin(centro_oeste), 'cust_Region'] = 'Midwest'
df_train.loc[df_train['customer_state'].isin(sul), 'cust_Region'] = 'South'


# In[73]:


cross_heatmap(df_train[df_train['price'] != -1], ['seller_state', 'cust_Region'], 
              values=df_train[df_train['price'] != -1]['freight_value'], aggfunc='mean')


# *INSIGHT*
# 
# *- SP sellers have a lowest mean of freights to all regions.*
# 
# *- CE sellers have high mean value*
# 
# *- We can note that the sellers from southeast could have a better competitive advantage.*

# ### What is the Payment Mode prefereed by the customers ? 

# In[74]:


df_train['price_log'] = np.log(df_train['price'] + 1.5)


# In[75]:


total = len(df_train)

plt.figure(figsize=(14,6))

plt.suptitle('Payment Type Distributions', fontsize=22)

plt.subplot(121)
g = sns.countplot(x='payment_type', data=df_train[df_train['payment_type'] != 'not_defined'])
g.set_title("Payment Type Count Distribution", fontsize=20)
g.set_xlabel("Payment Type Name", fontsize=17)
g.set_ylabel("Count", fontsize=17)

sizes = []
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
    
g.set_ylim(0, max(sizes) * 1.1)

plt.subplot(122)
g = sns.boxplot(x='payment_type', y='price_log', data=df_train[df_train['payment_type'] != 'not_defined'])
g.set_title("Payment Type by Price Distributions", fontsize=20)
g.set_xlabel("Payment Type Name", fontsize=17)
g.set_ylabel("Price(Log)", fontsize=17)

plt.subplots_adjust(hspace = 0.5, top = 0.8)

plt.show()


# *INSIGHTS - More than 73.5% of all sales are the Payment type is Credit Card. 
# Second most common Payment Type is "boleto"(invoice) with almost 19.5%. 
# The third more common payment type is voucher with 5.43%.
# We also have some payments from debit card and only 3 sales to not_defined*

# ## 2.2 Products Analysis

# By leveraging product data analysis, e-commerce businesses can make data-driven decisions, optimize their product offerings, and improve overall business performance.

# ### 2.2.1 Total Categories of Products Sold by Company

# In[76]:


print('Number of unique products categories sold by company are',len((product_filtered.product_category_name).unique()))


# ### 2.2.2 What are the Top 10 Categories sold by number

# In[77]:


plt.figure(figsize=(8,7))
idx = product_filtered['product_category_name'].value_counts()[:10].index
sns.countplot(y=product_filtered['product_category_name'], order=idx, palette="seismic")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Top categories', fontsize=14)
plt.title("Top 10 Categories sold", fontsize=20)
plt.show()


# *Insight : Top category sold is cama_mesa_banho followed by belza_saude*

# ### 2.2.3 What are Top 10 Selling  Product Categories by Total Revenue generated

# In[78]:


product_categorywise_cost= product_filtered.groupby('product_category_name')[['price']].sum().sort_values('price', ascending = False)


# In[79]:


product_categorywise_cost.head(10)


# In[80]:


product_categorywise_cost.head(10).plot(kind='barh', figsize=(12, 8))
plt.xlabel('Price in Million Brazilian Real', fontsize = 20)
plt.ylabel('Product Category Name', fontsize = 20)
plt. legend(fontsize = 20)
plt.xticks( fontsize = 16)
plt.yticks( fontsize = 16)
plt.title("Top 10 Selling Product Categories by Total Revenue generated", fontsize=20)


# *Insight- Though cama_mesa_banho has most of the product sold, belza_saude generates the higest total revenue*

# ### 2.2.4 What are the categories with costly items sold ? 
# #### Top 10 Product Categories by Average Price of an Item

# In[81]:


product_categorywise_avgcost= product_filtered.groupby('product_category_name')[['price']].mean().sort_values('price', ascending = False)
product_categorywise_avgcost.head(10)


# In[82]:


product_categorywise_avgcost.head(10).plot(kind='bar', figsize=(12, 8))
plt.xlabel('Product Category', fontsize = 20)
plt.ylabel('Average price in Real', fontsize = 20)
plt. legend(fontsize = 20)
plt.xticks( fontsize = 16)
plt.yticks( fontsize = 16)
plt.title("categories with costly items sold", fontsize=20)


# *Insight - Costly items sold on the platform are from pcs and portateis_casa_forno_e_cafe categories*

# ### 2.2.5 Top 10 Product Categories by Total Freight Cost 

# In[83]:


product_categorywise_freight= product_filtered.groupby('product_category_name')[['freight_value']].sum().sort_values('freight_value', ascending = False)


# In[84]:


product_categorywise_freight.head(10)


# In[85]:


product_categorywise_freight.head(15).plot(kind= 'bar', figsize=(12, 8))
plt.xlabel('Product Category', fontsize = 20)
plt.ylabel('Total Freight price in Real', fontsize = 20)
plt. legend(fontsize = 20)
plt.xticks( fontsize = 16)
plt.yticks( fontsize = 16)
plt.title("Top 10 Product Categories by Total Freight Cost", fontsize=20)


# *Insight : Highest total freight cost is from categories cama_mesa_banho followed by belza_saude*

# ### 2.2.6 Top 10 Product Categories by Average Freight Cost

# In[86]:


product_categorywise_avgfreight= product_filtered.groupby('product_category_name')[['freight_value']].mean().sort_values('freight_value', ascending = False)


# In[87]:


product_categorywise_avgfreight.head(10)


# In[88]:


product_categorywise_avgfreight.head(15).plot(kind = 'bar', figsize=(12, 8))
plt.xlabel('Product Category', fontsize = 20)
plt.ylabel('Average Freight price in Real', fontsize = 20)
plt. legend(fontsize = 20)
plt.xticks( fontsize = 16)
plt.yticks( fontsize = 16)
plt.title("Top 10 Product Categories by Average Freight Cost", fontsize=20)


# *Insight - Highest average freight cost are from pcs and portateis_casa_forno_e_cafe categories. Also note that these are  also the costly products*

# ### Average Freight State wise.... taking data from customer_filtered

# In[89]:


customer_filtered.head(10)


# In[90]:


statewise_avgfreight= customer_filtered.groupby('customer_state')[['freight_value']].mean().sort_values('freight_value', ascending = False)


# In[91]:


statewise_avgfreight.head(15).plot(kind = 'bar', figsize=(12, 8))
plt.xlabel('State', fontsize = 20)
plt.ylabel('Average Freight price in Real', fontsize = 20)
plt. legend(fontsize = 20)
plt.xticks( fontsize = 16)
plt.yticks( fontsize = 16)
plt.title("Average Freight State wise", fontsize=20)


# *Insight - Highest average freight cost are from states RR, PB and RO*

# ### Average Freight City wise.... taing data from customer_filtered

# In[92]:


citywise_avgfreight= customer_filtered.groupby('customer_city')[['freight_value']].mean().sort_values('freight_value', ascending = False)


# In[93]:


citywise_avgfreight.head(15).plot(kind = 'bar', figsize=(12, 8))
plt.xlabel('City', fontsize = 20)
plt.ylabel('Average Freight price in Real', fontsize = 20)
plt. legend(fontsize = 20)
plt.xticks( fontsize = 16)
plt.yticks( fontsize = 16)
plt.title("Average Freight City wise", fontsize=20)


# *Insight - At the city level, highest average freight cost are from cities itupiranga, amarante and almino afonso*

# ### Total Price of all goods

# In[94]:


product_filtered.price.sum()


# ### Total Freight Cost of All good

# In[95]:


product_filtered.freight_value.sum()


# ## 2.3 Customer Analysis

# In[96]:


customer_filtered.head(10)


# In[97]:


customer_filtered.describe()


# ### Top 5 repeat customers_id

# In[98]:


customer_filtered.customer_id.value_counts().head(5)


# ###  Top 10 repeated products

# In[99]:


customer_filtered.product_id.value_counts().head(10)


# ### 2.3.1 Cities with Highest Customer

# In[100]:


customer_filtered.customer_state.value_counts().head(20)


# In[101]:


customer_filtered.customer_city.value_counts().head(10).plot(kind ='barh', figsize=(12, 8));
plt.xlabel('Number of Customers', fontsize = 20)
plt.ylabel('Cities', fontsize = 20)
plt. legend(fontsize = 20)
plt.xticks( fontsize = 16)
plt.yticks( fontsize = 16)
plt.title("Cities with Highest Customer", fontsize=20)


# *INSIGHT - Sao paulo has highest number of customers. The second highest customers is in rio de janeiro which has less than half of the customers of  Sao Paulo*

# ### 2.3.2 States with Highest Customers 

# In[102]:


customer_filtered.customer_state.value_counts()


# In[103]:


customer_filtered.customer_state.value_counts().head(10).plot(kind ='barh', figsize=(12, 8));
plt.xlabel('Number of Customers', fontsize = 20)
plt.ylabel('States', fontsize = 20)
plt. legend(fontsize = 20)
plt.xticks( fontsize = 16)
plt.yticks( fontsize = 16)
plt.title("States with Highest Customers", fontsize=20)


# *INSIGHT - State of Sao paulo has highest number of customers. The second highest customers is in rio de janeiro which has nealry one thrid of the customers of  Sao Paulo*

# ![](https://www.researchgate.net/profile/Camila-Lorenz/publication/341914696/figure/fig1/AS:898814760198144@1591305546627/Map-of-Brazil-showing-the-states-belonging-to-each-region-Acronyms-for-each-state.jpg)

# ### 2.3.3 States with Lowest Customers

# In[104]:


customer_filtered.customer_state.value_counts().tail(10).plot(kind ='barh', figsize=(12, 8));
plt.xlabel('Number of Customers', fontsize = 20)
plt.ylabel('States', fontsize = 20)
plt. legend(fontsize = 20)
plt.xticks( fontsize = 16)
plt.yticks( fontsize = 16)
plt.title("States with Lowest Customers", fontsize=20)


# *INSIGHT: RR,AP and AC are the three states with the lowest number of customers; all below 100*

# ### CUSTOMER SEGMENTATION 
# 
# ![](https://images.datacamp.com/image/upload/v1659712759/Customer_Segmentation_ccf4cfac94.png)
# 
# ### 2.3.4 Recency

# The "Recency" is calculated by finding the difference between the maximum purchase date (MaxPurchaseDate) for each customer and the maximum purchase date overall in the dataset.

# In[105]:


#create a generic user dataframe to keep CustomerID and new segmentation scores
orders_user = pd.DataFrame(orders['customer_unique_id'].unique())
orders_user.columns = ['customer_unique_id']

#get the max purchase date for each customer and create a dataframe with it
orders_max_purchase = orders.groupby('customer_unique_id').order_purchase_timestamp.max().reset_index()
orders_max_purchase.columns = ['customer_unique_id','MaxPurchaseDate']

#we take our observation point as the max invoice date in our dataset
orders_max_purchase['Recency'] = (orders_max_purchase['MaxPurchaseDate'].max() - orders_max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
orders_user = pd.merge(orders_user, orders_max_purchase[['customer_unique_id','Recency']], on='customer_unique_id')

orders_user.head()


# In[106]:


#plot a recency histogram

plot_data = [
    go.Histogram(
        x=orders_user['Recency']
    )
]

plot_layout = go.Layout(
        title='Recency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


# *INSIGHT: Here we see that recency data is not visbilbe in the graph for 1-100 days. This is because data is not available for the last few months. Those time frame can be ignored. From the graph it is seen that number of orders arenealry uniform for that last 300 years with a high jump in orders some 325 days back may beacuse of some offer sale etc. Post that time the orders slowly drop with the recency till 625 days*

# In the context of customer segmentation, clustering can be useful for understanding and identifying different groups or segments of customers with similar behavior, preferences, or characteristics. By clustering customers based on their "Recency" values, we can potentially identify groups of customers who made purchases or interacted with a product/service recently, and those who did not.
# 
# The main need for this clustering code is to find an optimal number of clusters. The optimal number of clusters is often determined by finding the "elbow" point in the SSE plot. The SSE (sum of squared errors) is a measure of how closely the data points in each cluster are packed together. The elbow point represents a trade-off between having a low SSE value (tight clusters) and a high number of clusters (more granularity).
# 
# By visually inspecting the SSE plot, we can choose the number of clusters that provides a good balance between cluster tightness and the number of clusters we want to work with. This optimal number of clusters can then be used to further analyze and understand customer segments or for downstream tasks such as targeted marketing, personalized recommendations, or tailored customer experiences.

# In[107]:


from sklearn.cluster import KMeans

sse={}
orders_recency = orders_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(orders_recency)
    orders_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.title("Clustering Curve", fontsize=20)
plt.show()


# *INSIGHTS: By visually inspecting the SSE plot, we can choose the number of clusters that provides a good balance between cluster tightness and the number of clusters we want to work with. So based on the curve, we can choose 4 as the optimal number of cluster*
# 
# *This optimal number of clusters can then be used to further analyze and understand customer segments or for downstream tasks such as targeted marketing, personalized recommendations, or tailored customer experiences.*

# In[108]:


#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
kmeans.fit(orders_user[['Recency']])
orders_user['RecencyCluster'] = kmeans.predict(orders_user[['Recency']])

#function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

orders_user = order_cluster('RecencyCluster', 'Recency',orders_user,False)


# In[109]:


orders_user


# In[110]:


orders_user.groupby('RecencyCluster')['Recency'].describe()


# The purpose of the above analysis is to segment customers based on their recency of purchases or interactions. By creating clusters using the KMeans algorithm, we are grouping customers into four segments based on how recently they have made a purchase.

# By analyzing the different customer segments based on recency, businesses can make targeted decisions and strategies. For example, they can prioritize engagement efforts towards customers in the cluster with the highest recency, as they may be more likely to make a repeat purchase. On the other hand, they may want to implement re-engagement campaigns for customers in the cluster with the lowest recency to encourage them to make a purchase again.

# ### 2.3.5 Frequency

# Here 'Frequency' represents the order count for each customer. This information can be useful for analyzing customer behavior and making data-driven decisions. For example, businesses can identify customers who make frequent purchases and target them with special offers or discounts to encourage repeat purchases.

# In[111]:


#get order counts for each user and create a dataframe with it
orders_frequency = orders.groupby('customer_unique_id').order_purchase_timestamp.count().reset_index()
orders_frequency.columns = ['customer_unique_id','Frequency']

#add this data to our main dataframe
orders_user = pd.merge(orders_user, orders_frequency, on='customer_unique_id')
     


# In[112]:


#plot the histogram
plot_data = [
    go.Histogram(
        x=orders_user.query('Frequency < 1000')['Frequency']
    )
]

plot_layout = go.Layout(
        title='Frequency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


# *INSIGHT - From the above histogram, it is visible that most of the customers have placed only single order and the median order frequency is 1*

# In[113]:


#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(orders_user[['Frequency']])
orders_user['FrequencyCluster'] = kmeans.predict(orders_user[['Frequency']])

#order the frequency cluster
orders_user = order_cluster('FrequencyCluster', 'Frequency',orders_user,True)

#see details of each cluster
orders_user.groupby('FrequencyCluster')['Frequency'].describe()


#  The above code clusters customers into different groups based on their order frequencies. It assigns a cluster label to each customer and then orders the clusters based on the average order frequency within each cluster. The descriptive statistics provide additional insights into the distribution of order frequencies within each cluster.
# 
# This analysis can help businesses identify different customer segments based on their order frequencies and tailor their marketing strategies accordingly. For example, they can focus on retaining and upselling to customers in clusters with higher order frequencies or implement strategies to increase the order frequency of customers in clusters with lower frequencies.

# ### 2.3.6 Revenue for each Customer

# In[114]:


#calculate revenue for each customer
orders['Revenue'] = orders['price'] * orders['order_item_id']
orders_revenue = orders.groupby('customer_unique_id').Revenue.sum().reset_index()

#merge it with our main dataframe
orders_user = pd.merge(orders_user, orders_revenue, on='customer_unique_id')

#plot the histogram
plot_data = [
    go.Histogram(
        x=orders_user.query('Revenue < 10000')['Revenue']
    )
]

plot_layout = go.Layout(
        title='Monetary Value'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()
     


# *INSIGHTS - The median revenue generated by the customer is around 100 and customer count for higher revenue generated decreases almost exponentially*

# In[115]:


#apply clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(orders_user[['Revenue']])
orders_user['RevenueCluster'] = kmeans.predict(orders_user[['Revenue']])


#order the cluster numbers
orders_user = order_cluster('RevenueCluster', 'Revenue',orders_user,True)

#show details of the dataframe
orders_user.groupby('RevenueCluster')['Revenue'].describe()


# The above code clusters customers into different groups based on their revenue values. It assigns a cluster label to each customer and then orders the clusters based on the average revenue within each cluster. The descriptive statistics provide further insights into the distribution of revenue within each cluster.
# 
# This analysis can help businesses identify different customer segments based on their revenue and tailor their strategies accordingly. For example, they can target customers in clusters with higher revenue for personalized offers or implement strategies to increase the revenue of customers in clusters with lower revenue

# ### 2.3.7 Overall Segmentation based on Revenue, Frequency and Recency

# In[116]:


#calculate overall score and use mean() to see details
orders_user['OverallScore'] = orders_user['RecencyCluster'] + orders_user['FrequencyCluster'] + orders_user['RevenueCluster']
orders_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()


# In[117]:


orders_user['Segment'] = 'Low-Value'
orders_user.loc[orders_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
orders_user.loc[orders_user['OverallScore']>4,'Segment'] = 'High-Value' 


# ## 2.4 Seller Analysis

# In[118]:


print("Total number of unique seller_id:",len((seller_data.seller_id).unique()))


# ### 2.4.1 At which Location (city) the majority of sellers are located ?

# In[119]:


seller_data.head(5)


# In[120]:


seller_data.seller_state.value_counts().head(10)


# In[121]:


seller_data.seller_city.value_counts().head(10).plot(kind= 'barh', figsize=(12, 8))
plt.xlabel('Number of Sellers', fontsize = 20)
plt.ylabel('Cities', fontsize = 20)
plt. legend(fontsize = 20)
plt.xticks( fontsize = 16)
plt.yticks( fontsize = 16)
plt.title("city with majority of sellers", fontsize=20)


# *INSIGHT - Highest number of sellers are located in city of Sao Paulo. But the second highest are in curitiba and not Rio De Janerio. A root cause analysis needs to be done as to why the sale in Curitiba is so low inspite opf having high numbe rof seller*

# ### 2.4.2 Which states have high number of sellers ? 
# ### Location of Sellers State wise

# In[122]:


plt.figure(figsize=(16,12))

plt.subplot(212)
g = sns.countplot(x='seller_state', data=df_train, orient='h')
g.set_title("Seller's State Distribution", fontsize=20)
g.set_xlabel("State Name Short", fontsize=17)
g.set_ylabel("Count", fontsize=17)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
sizes = []
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=12) 
g.set_ylim(0, max(sizes) * 1.1)


# *INSIGHT - Among States, highest number of sellers are located in Sao Paulo followed by MG and PR but Rio De Janerio has higher number of sales than MG and PR. This may be possible that if Rio can get more number of sellers then the sale can even increase further*

# In[123]:


plt.figure(figsize=(16,12))

plt.suptitle('SELLER State Distributions', fontsize=22)



plt.subplot(221)
g2 = sns.boxplot(x='seller_state', y='price_log', 
                 data=df_train[df_train['price'] != -1])
g2.set_title("Seller's State by Price", fontsize=20)
g2.set_xlabel("State Name Short", fontsize=17)
g2.set_ylabel("Price(Log)", fontsize=17)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)

plt.subplot(222)
g3 = sns.boxplot(x='seller_state', y='freight_value', 
                 data=df_train[df_train['price'] != -1])
g3.set_title("Seller's State by Freight Value", fontsize=20)
g3.set_xlabel("State Name Short", fontsize=17)
g3.set_ylabel("Freight Value", fontsize=17)
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)

plt.subplots_adjust(hspace = 0.5, top = 0.9)

plt.show()


# *INSIGHT - SP has the lowest median log(Price) while BA, AC and MS have the highest. Median Freight value also lopwest for SP and increases as we move towards other state*

# ## 2.5 Reviews Analysis

# Scores Count 

# In[124]:


reviews_filtered.review_score.value_counts()


# Classifying scores less than or equal to 3 as unsatisfactory and above 3 as Satisfactory

# In[125]:


def partition(x):
    if x < 3:
        return 0
    return 1
reviews_filtered['review_score']=reviews_filtered['review_score'].map(lambda cw : partition(cw) ) 
    
# checking the review score now
reviews_filtered.review_score.value_counts()


# In[126]:


reviews_filtered.head(5)


# In[127]:


new_reviews = reviews_filtered

new_reviews= new_reviews.replace(1,"Satisfactory")
new_reviews = new_reviews.replace(0,"Unsatisfactory")

new_reviews.sample(10)


# In[128]:


new_reviews.review_score.value_counts()


# In[129]:


# plt.suptitle('Reviews Distributions', fontsize=22)
new_reviews.review_score.value_counts().plot.pie(shadow= True, figsize=(6,6));
plt.title("Reviews Division", fontsize=20)


# *INSIGHT - Nealry 12.5% of overall customers who posted review are unstatisfied with service*

# ### What is the Statewise Performance Based on Customer Reviews ?

# In[130]:


review_statewise = new_reviews.groupby(by = ['customer_state', 'review_score']).count().sort_values('customer_state',ascending = False)


# In[131]:


review_statewise = review_statewise.reset_index()


# In[132]:


review_statewise.head()


# In[133]:


plt.figure(figsize=(12,7))
sns.barplot(x='customer_state', y ='order_id', hue ='review_score', data = review_statewise.head(100) )

plt.xticks(fontsize=13)
plt.yticks(fontsize=20)
plt.xlabel('States', fontsize=22)
plt.ylabel('Reviews', fontsize=22)
plt.title("State wise positive and negative reviews ", fontsize=20)
plt.show()


# *INSIGHT -  15% of customers have reported unstatisfactory performance which majority of which are from Southeast region with SP having 10% unsatisfied, RJ having 21% unsatisfies, and MG with 15% unsatisfied*

# # Insights and Suggestions

# • Monthly Revenue grew consistently from 2016 till November 2017 and stagnated since then. New Strategies need to be formulated to grow the sales
# 
# • Monthly growth rate is reducing consistently since 2018. A comparative Analysis is needed with respect to other competitors to understand the reasons for the slowdown in growth
# 
# • Monthly Active customers have also stagnated since 2018. New Marketing starategies and Markets can be tapped. Steps to improve quality of service to increase repeat customers need to be taken
# 
# • Growth of existing users is very low and majority of customers are new customers only
# 
# •This also means customer retention is very low hovering near 0.5%
# 
# • Huge Difference between expected delivery time and actual delivery time is one of the major reason for poor customer experience and lack of repeat customers
# 
# • SP has lowest delivery rates while other states CE have the highest delivery rates accross all regions. Thus a comparitive study and comparison between distributor of SP and other states can be made to improve the distribution channels of other states
# 
# • Sellers from south east have better competitive advantage if we see the delivery price
# 
# • More than 73.5% of all sales are the Payment type is Credit Card. Second most common Payment Type is "boleto"(invoice) with almost 19.5%.
# 
# • beleza_saude, relogios_presentes, cama_mesa_banho are the top three categories in terms of revenue generated
# 
# • pcs, portateis_casa_forno_e_cafe, eletrodomesticos_2 are the top three categories by average value of products sold
# 
# • pcs, moveis_colchao_e_estofado, eletrodomesticos_2, categories cost the most in average delivery charge
# 
# • SP, RJ, MG are the cities with highest customers and higest number of sales
# 
# • From the frequency cluster analysis, nealry 84% of customers order only once, some 14% order for more than two times and 1.9% nearly 5 times and around 0.1% more than 5 times
# 
# • Based on above cluster analysis, focus on retaining and upselling to customers in clusters with higher order frequencies or implement strategies to increase the order frequency of customers in clusters with lower frequencies.
# 
# • Most of the sellers located in SP which is the main reason for high sale in SP as well as better delivery performance as well
# 
# • 15% of customers have reported unstatisfactory performance which majority of which are from Southeast region with SP having 10% unsatisfied, RJ having 21% unsatisfies, and MG with 15% unsatisfied **
# 

import streamlit as st
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import calendar
from func import DataFrameProcess


# Dataset
main_df = pd.read_csv('./dashboard/clean_data.csv')

# Change column to datetime
main_df['order_purchase_timestamp'] = pd.to_datetime(main_df['order_purchase_timestamp'])

# Read the Dataset
function_read_df = DataFrameProcess(main_df)

# Datasets
average_spend_df = function_read_df.create_avg_spend_df()
month_most_spend_df, month_most_cust_df = function_read_df.create_month_spend()
geo_spend = function_read_df.create_geo_spend_df()
average_time_purchase = function_read_df.create_avg_time_purchase_df()
rfm_score = function_read_df.create_rfm_df()


# Title Dashboard
st.markdown(
    "<h1 style='text-align: left;'>E-Commerce Dashboard</h1>"
    "<p> Summary Sales in 2017</p>", 
    unsafe_allow_html=True
)


# Create SideBar
with st.sidebar:
    # Title
    st.markdown("<h1 style='font-size: 20px;'>Muhammad Adi Nugroho</h1>", unsafe_allow_html=True)
 
# Summary Info
col1, col2, col3 = st.columns(3)

with col1:
    total_order = main_df['order_id'].count()
    st.info(f'Success Transaction: **{total_order}**')

with col2:
    total_revenue = main_df['payment_value'].sum()
    st.info(f'Total Revenue: **{total_revenue}**')

with col3:
    average_spending = np.median(average_spend_df['payment_value'])
    st.info(f'Average Purchases: **{average_spending}**')

# Monthly Spend and Customers
st.markdown(
    "<h2 style='text-align:left; font-size: 20px;'>Customer Spend Money</h2>", 
    unsafe_allow_html=True    
)

month_most_spend_df['Month'] = [calendar.month_name[m] for m in month_most_spend_df['order_purchase_timestamp']]
max_month = month_most_spend_df.loc[month_most_spend_df['payment_value'].idxmax()]['Month']

col1, col2 = st.columns(2)
with col1:
    f'Highest Spend\t: **{max_month}**'

with col2:
    st.markdown(
        f"<div style='text-align: right;'>Total Spend : <b>{month_most_spend_df['payment_value'].max()}</b></div>", 
        unsafe_allow_html=True
    )
    
# Monthly Spend
plt.figure(figsize=(12, 6))
sns.lineplot(month_most_spend_df, x='Month', y='payment_value')
plt.xlabel('')
plt.ylabel('Spending')
st.pyplot(plt)

st.markdown(
    "<h2 style='text-align:left; font-size: 20px;'>Orders</h2>", 
    unsafe_allow_html=True
)

# Monthly Customer
month_most_cust_df['Month'] = [calendar.month_name[m] for m in month_most_cust_df['order_purchase_timestamp']]
max_monthc = month_most_cust_df.loc[month_most_cust_df['order_id'].idxmax()]['Month']

col1, col2 = st.columns(2)
with col1:
    f'Highest Orders\t: **{max_month}**'

with col2:
    st.markdown(
        f"<div style='text-align:right;'>Total Orders : <b>{month_most_cust_df['order_id'].max()}</b></div>",
        unsafe_allow_html = True
    )

plt.figure(figsize=(12, 6))
sns.lineplot(month_most_cust_df, x='Month', y='order_id')
plt.xlabel('')
plt.ylabel('Customer Orders')
st.pyplot(plt)

# Geolocation 
st.markdown(
    "<h2 style='text-align: left; font-size: 20px;'> Top 5 City by Revenue</h2>",
    unsafe_allow_html=True
)

plt.figure(figsize=(12, 6))
sns.barplot(geo_spend, x=geo_spend['customer_city'], y=geo_spend['payment_value'])

plt.xlabel('')
plt.ylabel('Total Revenue')
st.pyplot(plt)

# RFM
st.markdown(
    "<h2 style='text-align:left; font-size: 20px;'>Customers Segmentation by RFM Score</h2>", 
    unsafe_allow_html=True  
)

# Create RFM Classification
rfm_score_vis = rfm_score['Customer_Classification'].value_counts().reset_index()

col1, col2 = st.columns(2)

with col1:
    f"Total High Value Customers : **{rfm_score_vis['count'].max()}**"

plt.figure(figsize=(12, 4))

plt.pie(rfm_score_vis['count'], 
        labels=rfm_score_vis['Customer_Classification'],
        autopct='%1.1f%%', startangle=140,
        explode = (0, 0.1),
        colors=['grey', 'red'])

st.pyplot(plt)
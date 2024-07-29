import pandas as pd
import numpy as np
import datetime

class DataFrameProcess:
    def __init__ (self, df):
        self.df = df


    def create_avg_spend_df(self):
        
        avg_spend_df = self.df.groupby('customer_unique_id')['payment_value'].mean().reset_index()
        
        return avg_spend_df
    

    def create_month_spend(self):
        
        month_most_spend_df = self.df.groupby(self.df['order_purchase_timestamp'].dt.month)\
            ['payment_value'].sum().reset_index()

        month_most_cust_df = self.df.groupby(self.df['order_purchase_timestamp'].dt.month)\
            ['order_id'].count().reset_index()

        return month_most_spend_df, month_most_cust_df
    

    def create_geo_spend_df(self):
        
        geo_spend_df = self.df.groupby('customer_city')['payment_value'].sum().reset_index()
        geo_spend_df = geo_spend_df.sort_values(by='payment_value', ascending=False).head(5)
        
        return geo_spend_df
    
    
    def create_avg_time_purchase_df(self):
        
        avg_time_purchase_df = self.df[['customer_unique_id', 'order_purchase_timestamp']]\
            .sort_values(by=['customer_unique_id', 'order_purchase_timestamp'])
        
        avg_time_purchase_df['time_diff'] = avg_time_purchase_df.groupby('customer_unique_id') \
            ['order_purchase_timestamp'].diff()
        
        avg_time_purchase_df['time_days_diff'] = avg_time_purchase_df['time_diff'] / pd.Timedelta(days=1)
        isnotna_avg_time_df = avg_time_purchase_df.loc[~avg_time_purchase_df.isna().any(axis=1)]
        
        return isnotna_avg_time_df
    
    def create_rfm_df(self):
        rfm_data = self.df.groupby('customer_unique_id').size().reset_index(name='frequency')
        latest_date = self.df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
        monetary = self.df.groupby('customer_unique_id')['payment_value'].sum().reset_index()
        
        rfm_data = pd.merge(
            rfm_data,
            latest_date,
            on='customer_unique_id'
        )
        
        rfm_data = pd.merge(
            rfm_data,
            monetary,
            on='customer_unique_id',
            how='left'
        )

        current_date = pd.to_datetime('2018-01-01')
        rfm_data['recency'] = (current_date - rfm_data['order_purchase_timestamp']).dt.days

        percentiles = [0, 20, 40, 60, 80]

        def assign_score_by_percentile(value, thresholds, reverse=False):
            if reverse:
                value = max(value, thresholds[0])  # Fixed referencing parameters instead of self
                for i, threshold in enumerate(thresholds):
                    if value <= threshold:
                        return len(thresholds) - i  # Reverse the score
                return 1
            else:
                for i, threshold in enumerate(thresholds):
                    if value <= threshold:
                        return i + 1
                return len(thresholds) + 1
        
        recency_thresholds = np.percentile(rfm_data['recency'], percentiles)
        rfm_data['R'] = rfm_data['recency'].apply(lambda x: assign_score_by_percentile(x, recency_thresholds, reverse=True))

        frequency_thresholds = np.percentile(rfm_data['frequency'], percentiles)
        rfm_data['F'] = rfm_data['frequency'].apply(lambda x: assign_score_by_percentile(x, frequency_thresholds))*100

        payment_value_thresholds = np.percentile(rfm_data['payment_value'], percentiles)
        rfm_data['M'] = rfm_data['payment_value'].apply(lambda x: assign_score_by_percentile(x, payment_value_thresholds))*10

        rfm_data['RFM Score'] = rfm_data['F'] + rfm_data['M'] + rfm_data['R']

        def map_rfm_score(rfm_score):
            if rfm_score >= 500:
                return 'High Value Customer'
            elif  300 < rfm_score < 500:
                return 'Mid Value Customer'
            else:
                return 'Low Value Customer'

        rfm_data['Customer_Classification'] =  rfm_data['RFM Score'].apply(lambda x: map_rfm_score(x))

        return rfm_data

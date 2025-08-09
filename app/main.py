from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np


#simple categorization logc
def categorize(description):
    desc = description.lower()
    if any(word in desc for word in ['uber', 'taxi', 'bus']):
        return 'Transport'
    elif any(word in desc for word in['grocery', 'food', 'supermarket']):
        return 'Groceries'
    elif any(word in desc for word in ['netflix', 'prime', 'entertainment']):
        return 'Entertainment'
    elif 'salary' in desc:
        return 'Income'
    elif any(word in desc for word in ['coffee', 'cafe']):
        return 'Dining'
    elif any(word in desc for word in['amazon', 'shopping']):
        return 'Shopping'
    else:
        return 'Other'
    
st.set_page_config(page_title="Personal Finance Dashboard", layout="wide")
st.title("💰 Personal Finance Dashboard")
st.write("✅ App is running...")  # Add this line to confirm code is executing

uploaded_file = st.file_uploader("📤 Upload your bank statement CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()
    df['Category'] = df['description'].apply(categorize)
    st.subheader("📊 Categorized Transactions")
    df['date'] = pd.to_datetime(df['date'])
    st.dataframe(df)
    
    #Group by date is in datetime format
    daily_summary = df.groupby(df['date'].dt.day).agg({'amount':'sum'}).reset_index()
    daily_summary.columns = ['day', 'net_amount']
    total_income = df[df['amount']>0]['amount'].sum()
    total_spent = df[df['amount']<0]['amount'].sum()

    col1, col2 = st.columns(2)
    col1.metric('Total Income', f"${total_income:,.2f}")
    col2.metric('Total Spent', f"${-total_spent:,.2f}")

    st.subheader("📈 Spending by Category")
    st.bar_chart(df[df["amount"]<0].groupby('Category')["amount"].sum().abs())


    #Prepare data 
    x = daily_summary[['day']]
    y = daily_summary['net_amount'].cumsum() #cumulative savings trend

    #Train model
    model = LinearRegression()
    model.fit(x,y)

    #Predict for full month
    future_days = np.arange(1,31).reshape(-1, 1)
    predicted_savings = model.predict(future_days)

    final_savings = predicted_savings[-1]

    st.subheader("💸 Predicted Month-End Savings")
    st.metric("Expected savings by Day 30", f'${final_savings:,.2f}')

    #plot
    plt.figure(figsize=(10,4))
    plt.plot(future_days, predicted_savings, label = 'Predicted Savings Trend', color= 'Teal')
    plt.scatter(x, y, color='blue', label='Actual')
    plt.axhline(0, color = 'red', linestyle = '--')
    plt.xlabel('Day of Month')
    plt.ylabel('Cumulative Savings')
    plt.legend()
    st.pyplot(plt)

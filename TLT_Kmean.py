import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set Streamlit page config
st.set_page_config(page_title="TLT Monthly Return Clustering", layout="wide", initial_sidebar_state="auto")

# Title
st.title("K-Means Clustering of TLT Monthly Returns")

# Download data
tlt = yf.download("TLT", period="10y", interval="1d")
price_col = 'Adj Close' if 'Adj Close' in tlt.columns else 'Close'
monthly_prices = tlt[price_col].resample('ME').last()

# Monthly returns
monthly_returns = monthly_prices.pct_change().dropna() * 100
monthly_returns_df = pd.DataFrame(monthly_returns)
monthly_returns_df.columns = ['Return']
monthly_returns_df['Month'] = monthly_returns_df.index.month
monthly_returns_df['MonthName'] = monthly_returns_df.index.month_name()
monthly_returns_df['Year'] = monthly_returns_df.index.year

# Monthly averages
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_avg = monthly_returns_df.groupby('MonthName')['Return'].mean().reindex(month_order).fillna(0)
monthly_avg_values = monthly_avg.values.reshape(-1, 1)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(monthly_avg_values)

# Color coding
colors = ['green' if val >= 0 else 'red' for val in monthly_avg.values]

# Most recent month info
most_recent_month = monthly_returns_df.index[-1].month_name()
most_recent_return = monthly_returns_df.iloc[-1]['Return']
historical_avg = monthly_avg[most_recent_month]
difference = most_recent_return - historical_avg

# Subtitle string
subtitle_str = ', '.join([
    f"{month[:3]}: {val:.2f}%" for month, val in zip(monthly_avg.index, monthly_avg.values)
])

# Plot
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
bars = ax.bar(monthly_avg.index, monthly_avg.values, color=colors)
ax.axhline(0, color='white', linewidth=0.8)

recent_idx = monthly_avg.index.tolist().index(most_recent_month)
bars[recent_idx].set_edgecolor('white')
bars[recent_idx].set_linewidth(2)
ax.text(recent_idx, monthly_avg[recent_idx] + 0.2,
        f"{most_recent_return:.2f}%\n({'+' if difference > 0 else ''}{difference:.2f}%)",
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

ax.set_title("K-Means Clustering of Average Monthly Returns for TLT (Last 10 Years)",
             color='white', fontsize=16, pad=20)
ax.set_xlabel(subtitle_str, color='white', fontsize=10, labelpad=20)
ax.set_ylabel("Average Return (%)", color='white')
ax.tick_params(colors='white')
plt.xticks(rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.3, color='white')
plt.tight_layout()

# Show in Streamlit
st.pyplot(fig)

# Optional: Display data tables
with st.expander("Show Monthly Average Returns"):
    st.dataframe(monthly_avg.round(2))

with st.expander("Show Raw Monthly Returns"):
    st.dataframe(monthly_returns_df[['Return', 'MonthName', 'Year']].sort_index(ascending=False))

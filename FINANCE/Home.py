# Home.py
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="FinBase",
    page_icon="💹",
    layout="wide"
)

# Title and description
st.title("💹 FinBase - The Base of All Your Financial Planning")
st.write("Welcome to FinBase! Explore the following tools to manage your finances and investments.")

# App descriptions
st.subheader("Available Apps")
st.write("""
1. **Stock Trading Bot**: Predict future stock prices using historical data and deep learning models.
2. **Crypto Trading Bot**: Forecast cryptocurrency prices based on market trends and historical data.
3. **Multilanguage Financial Document Analyzer**: Extract and analyze financial documents in multiple languages.
4. **Currency Converter**: Convert between currencies using real-time exchange rates.
5. **Financial Advisor**: Helps you to set your savings goals, and generate insightful financial reports for better money management.
6. **Expense Tracker**: Helps you record, monitor, and analyze your daily expenses to manage your finances effectively.
7. **Mutual Funds Analyzer**: Analyze mutual funds scheme, generate detailed dashboards for each scheme.                     
""")

# Navigation
st.sidebar.title("Navigation")
st.sidebar.write("Select an app from the sidebar to get started.")

# Footer
st.write("---")
st.write("Made with ❤️ by Amritanshu Bhardwaj")
st.write("© 2025 FinBase. All rights reserved.")
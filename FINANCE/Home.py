# Home.py
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Finance App Suite",
    page_icon="💹",
    layout="wide"
)

# Title and description
st.title("💹 Finance App Suite")
st.write("Welcome to the Finance App Suite! Explore the following tools to manage your finances and investments.")

# App descriptions
st.subheader("Available Apps")
st.write("""
1. **Stock Price Predictor**: Predict future stock prices using historical data and deep learning models.
2. **Crypto Price Predictor**: Forecast cryptocurrency prices based on market trends and historical data.
3. **Multilanguage Invoice Extractor**: Extract and analyze invoice data from documents in multiple languages.
4. **Currency Converter**: Convert between currencies using real-time exchange rates.
""")

# Navigation
st.sidebar.title("Navigation")
st.sidebar.write("Select an app from the sidebar to get started.")

# Footer
st.write("---")
st.write("Made with ❤️ by Amritanshu Bhardwaj")
st.write("© 2025 Finance App Suite. All rights reserved.")
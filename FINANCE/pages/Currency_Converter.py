import streamlit as st
import requests
import google.generativeai as genai

# Set up Google Gemini API
GENAI_API_KEY = "GOOGLE_API_KEY"
genai.configure(api_key=GENAI_API_KEY)

# Exchange rate API key
EXCHANGE_RATE_API_KEY = "b11799c6346665da9664fec1"
EXCHANGE_API_URL = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/latest/"

# Function to get exchange rates
def get_exchange_rate(from_currency, to_currency):
    url = EXCHANGE_API_URL + from_currency
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200 and "conversion_rates" in data:
        rates = data["conversion_rates"]
        return rates.get(to_currency, None)
    else:
        return None

# Function to convert currency
def convert_currency(amount, from_currency, to_currency):
    rate = get_exchange_rate(from_currency, to_currency)
    if rate:
        return round(amount * rate, 2)
    return None

# Function to use Gemini AI for natural language conversion
def ai_currency_conversion(query):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Convert currency: {query}")
    return response.text.strip()

# Streamlit UI
st.title("💱 Currency Converter with Gemini AI")

# User Input
query = st.text_input("Enter conversion query (e.g., 'Convert 100 USD to EUR')")

if st.button("Convert with AI"):
    if query:
        result = ai_currency_conversion(query)
        st.success(f"AI Result: {result}")
    else:
        st.warning("Please enter a conversion query!")

st.markdown("---")

st.subheader("Manual Currency Converter")
amount = st.number_input("Enter amount:", min_value=0.01, format="%.2f")
from_currency = st.selectbox("From Currency", ["USD", "EUR", "GBP", "INR", "JPY", "AUD"])
to_currency = st.selectbox("To Currency", ["USD", "EUR", "GBP", "INR", "JPY", "AUD"])

if st.button("Convert Manually"):
    converted_amount = convert_currency(amount, from_currency, to_currency)
    if converted_amount is not None:
        st.success(f"{amount} {from_currency} = {converted_amount} {to_currency}")
    else:
        st.error("Conversion failed! Please check the currency codes.")

st.markdown("🔹 Powered by Google Gemini & Exchange Rate API")

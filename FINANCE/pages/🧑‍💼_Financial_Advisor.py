import streamlit as st
from google import genai
from decouple import config
import plotly.express as px
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Financial Advisor",layout="wide",page_icon="🧑‍💼")
# Configure Gemini API
client = genai.Client(api_key="AIzaSyBWc7Ym6qMSo04uD-KtfT1JSin5AqhtyNg")

st.title("Smart Finance Advisor")
st.write("Get personalized financial advice based on your income, expenses, and goals.")

# --- USER INPUTS ---
st.header("Enter Your Financial Information")

# Basic Info
age = st.number_input("Age", min_value=18, max_value=100, step=1)
income = st.number_input("Monthly Income (Rs)", min_value=0)
savings = st.number_input("Total Current Savings (Rs)", min_value=0)
goal = st.selectbox("Your Financial Goal", ["Buy a Car", "Buy a House", "Retirement Planning", "Wealth Creation", "Emergency Fund"])
risk = st.selectbox("Risk Appetite", ["Low", "Medium", "High"])

# Expenses
st.subheader("Monthly Expense Breakdown")
rent = st.number_input("Rent/Mortgage", min_value=0)
utilities = st.number_input("Utilities (Electricity, Water, etc.)", min_value=0)
groceries = st.number_input("Groceries", min_value=0)
transport = st.number_input("Transportation", min_value=0)
subscriptions = st.number_input("Subscriptions (Netflix, Spotify, etc.)", min_value=0)
insurance = st.number_input("Health/Insurance", min_value=0)
education = st.number_input("Education/EMIs", min_value=0)
entertainment = st.number_input("Entertainment/Dining Out", min_value=0)
shopping = st.number_input("Shopping(Clothing/Accesories etc)", min_value=0)
others = st.number_input("Other Expenses", min_value=0)

# Button to trigger advice
if st.button("Get Financial Advice"):
    total_expenses = rent + utilities + groceries + transport + subscriptions + insurance + education + entertainment + shopping + others
    disposable_income = income - total_expenses

    # Create the prompt
    prompt = f"""
    You are a smart financial advisor. Here's the user's financial profile:
    - Age: {age}
    - Monthly Income: Rs{income}
    - Total Savings: Rs{savings}
    - Monthly Expenses:
        - Rent: Rs{rent}
        - Utilities: Rs{utilities}
        - Groceries: Rs{groceries}
        - Transportation: Rs{transport}
        - Subscriptions: Rs{subscriptions}
        - Insurance: Rs{insurance}
        - Education: Rs{education}
        - Entertainment: Rs{entertainment}
        - Shopping: Rs{shopping}
        - Other: Rs{others}
    - Financial Goal: {goal}
    - Risk Appetite: {risk}
    
    Total Monthly Expenses: Rs{total_expenses}
    Disposable Income: Rs{disposable_income}

    Based on this, give personalized financial advice. Include:
    - Budget optimization tips
    - Ideal monthly saving strategy
    - Investment suggestions (based on risk level)
    - Timeline to achieve financial goal
    """

    # Call Gemini
    response = client.models.generate_content(
        model= "gemini-2.5-flash-preview-04-17",
        contents= prompt
    )
    advice = response.text

    st.header("Your Personalized Financial Advice")
    st.markdown(advice)
    
     # --- Financial Summary Dashboard ---
    st.subheader("Monthly Financial Summary")
    labels = ['Rent', 'Utilities', 'Groceries', 'Transport', 'Subscriptions', 'Insurance', 'Education', 'Entertainment', 'Shopping', 'Other', 'Savings']
    values = [rent, utilities, groceries, transport, subscriptions, insurance, education, entertainment, shopping, others, disposable_income]

    fig = px.pie(
    names=labels,
    values=values,
    title="Pie Chart",
    hole=0,  # set to >0 if you want a donut chart
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Total Expenses:** Rs{total_expenses}")
    st.markdown(f"**Disposable Income (Potential Savings):** Rs{disposable_income}")

    # --- PDF Report Generation ---
    st.subheader("Download Financial Report")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Smart Finance Advisor Report", align='C')
    pdf.ln(5)

    pdf.multi_cell(0, 10, f"Age: {age}\nMonthly Income: Rs{income}\nTotal Savings: Rs{savings}\nFinancial Goal: {goal}\nRisk Appetite: {risk}\n")
    pdf.multi_cell(0, 10, f"Monthly Expenses:\nRent: Rs{rent}\nUtilities: Rs{utilities}\nGroceries: Rs{groceries}\nTransport: Rs{transport}\nSubscriptions: Rs{subscriptions}\nInsurance: Rs{insurance}\nEducation: Rs{education}\nEntertainment: Rs{entertainment}\nShopping: Rs{shopping}\nOther: Rs{others}\n\nTotal Expenses: Rs{total_expenses}\nDisposable Income: Rs{disposable_income}\n")
    pdf.multi_cell(0, 10, "Financial Advice:\n" + advice)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        st.download_button("Download PDF Report", data=open(tmp.name, "rb"), file_name="financial_report.pdf", mime="application/pdf")
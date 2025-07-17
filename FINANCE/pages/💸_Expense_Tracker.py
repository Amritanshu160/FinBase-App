import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from fpdf import FPDF
import plotly.express as px
from io import BytesIO
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="Expense Tracker",layout="wide",page_icon="💸")

# CSV file to store expenses
CSV_FILE = "expenses.csv"

# Initialize CSV if not present
def init_csv():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["Date", "Category", "Amount", "Description"])
        df.to_csv(CSV_FILE, index=False)

# Load expenses
def load_expenses():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        # Convert Date column to datetime, then to date object (no time)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        return df
    return pd.DataFrame(columns=["Date", "Category", "Amount", "Description"])

# Add expense
def add_expense(date, category, amount, description):
    df = load_expenses()
    # Ensure date is in YYYY-MM-DD format (no time component)
    date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
    new_row = {"Date": date_str, "Category": category, "Amount": amount, "Description": description}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)



# Clean up CSV data - NEW FEATURE
def clean_up_csv(complete_reset=False):
    # Option to completely reset the file
    if complete_reset:
        # Create a fresh empty CSV with only headers
        df = pd.DataFrame(columns=["Date", "Category", "Amount", "Description"])
        df.to_csv(CSV_FILE, index=False)
        return "CSV file completely reset. All data has been removed."
    
    # Regular cleanup
    df = load_expenses()
    if df.empty:
        return "No data to clean."
    
    original_count = len(df)
    
    # Remove rows with missing important values
    df = df.dropna(subset=["Date", "Amount", "Category"])
    
    # Fix date format issues
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])  # Remove rows with invalid dates
    
    # Convert amount to numeric
    df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce')
    df = df[df["Amount"] > 0]  # Remove rows with invalid or negative amounts
    
    # Ensure Description is a string
    df["Description"] = df["Description"].astype(str)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    df.to_csv(CSV_FILE, index=False)
    
    cleaned_count = len(df)
    removed = original_count - cleaned_count
    
    return f"Cleanup complete. Removed {removed} problematic entries."


# Plot category-wise pie chart with Plotly
def plot_pie(data):
    pie_data = data.groupby("Category")["Amount"].sum().reset_index()
    fig = px.pie(
        pie_data,
        names="Category",
        values="Amount",
        title="Category-wise Expense Distribution",
        hole=0  # set to 0.4 for donut chart if needed
    )
    st.plotly_chart(fig, use_container_width=True)


# Streamlit App
st.title("Personal Finance Manager")

init_csv()

menu = st.sidebar.selectbox("Menu", ["Add Expense", "Dashboard", "View All Data", "Clean Data"])

if menu == "Add Expense":
    st.header("Add New Expense")
    date = st.date_input("Date", datetime.now().date())
    category = st.selectbox("Category", ["Rent", "Utilities", "Groceries", "Transportation", "Subscriptions", "Insurance", "Education", "Entertainment", "Shopping", "Others"])
    amount = st.number_input("Amount (INR)", min_value=0.0, format="%.2f")
    description = st.text_input("Description")

    if st.button("Add Expense"):
        add_expense(date, category, amount, description)
        st.success("Expense added successfully!")

elif menu == "Dashboard":
    st.header("Expense Dashboard")
    df = load_expenses()
    if df.empty:
        st.info("No expenses recorded yet.")
    else:
        st.subheader("Category-wise Distribution")
        plot_pie(df)

        st.subheader("Summary")
        st.write(df.groupby("Category")["Amount"].sum().sort_values(ascending=False))

        st.write(f"**Total Spent**: Rs. {df['Amount'].sum():.2f}")

elif menu == "Clean Data":
    st.header("Reset All Expense Data")
    
    st.error("⚠️ WARNING: This will DELETE ALL expense data. The CSV file will be reset to 0 rows with only headers.")
    st.warning("This action CANNOT be undone!")
    
    confirm_reset = st.text_input("Type 'DELETE ALL DATA' to confirm complete reset:")
    
    if st.button("Reset All Data", key="reset_data_button"):
        if confirm_reset == "DELETE ALL DATA":
            result = clean_up_csv(complete_reset=True)
            st.success(result)
        else:
            st.error("Confirmation text doesn't match. Data was not deleted.")

elif menu == "View All Data":
    st.header("Complete Expense Records")
    df = load_expenses()
    
    if not df.empty:
        # Show total records and summary stats
        st.write(f"Total Records: {len(df)}")
        st.write(f"Total Amount: ₹{df['Amount'].sum():.2f}")
        
        # Display the full dataframe with improved formatting
        st.dataframe(
            df.style.format({
                'Amount': '₹{:.2f}',
                'Date': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
            }),
            height=600,
            use_container_width=True
        )
    else:
        st.info("No expense records found.")                

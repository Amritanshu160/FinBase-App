from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
from google import genai
import pathlib

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get Gemini response
def get_gemini_response(input_prompt, file_data, user_input=None):
    # If file is not an image (PDF), upload it first
    if not isinstance(file_data, Image.Image):
        # Determine MIME type based on file extension
        file_extension = pathlib.Path(uploaded_file.name).suffix.lower()
        mime_type = {
            '.pdf': 'application/pdf'
        }.get(file_extension, 'application/octet-stream')
        
        # Upload the file
        uploaded_file_obj = client.files.upload(
            file=file_data,
            config=dict(mime_type=mime_type)
        )
        contents = [input_prompt, uploaded_file_obj]
        if user_input:  # Only add user input if it exists
            contents.append(user_input)
    else:
        contents = [input_prompt, file_data]
        if user_input:  # Only add user input if it exists
            contents.append(user_input)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )
    return response.text

# Streamlit App Configuration
st.set_page_config(page_title="Multilanguage Financial Document Analyzer", layout="wide", page_icon="📃")
st.header("📃 Multilanguage Financial Document Analyzer")

# Chat memory initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! Upload a financial document or image and I'll help you understand or answer questions about it."}
    ]

# File upload section
uploaded_file = st.file_uploader("Upload a financial document (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])

# Display file preview
file_data = None
if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        file_data = Image.open(uploaded_file)
        st.image(file_data, caption="Uploaded Image", use_column_width=True)
    else:
        st.success(f"Uploaded file: {uploaded_file.name}")
        file_data = uploaded_file

# Submit button for initial analysis
submit = st.button("Analyze")

# Gemini prompt for document analysis
input_prompt = """
You are an expert in understanding financial documents like invoices, bank statements, balance sheets, etc.
A user will upload an image or document, and you'll be asked to analyze or answer questions based on it.
"""

# Initial analysis response (without user input)
if submit and uploaded_file is not None:
    response = get_gemini_response(input_prompt, file_data)
    # Add to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Chat UI for Q&A
st.divider()
st.subheader("💬 Ask Questions About the Document")

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your uploaded document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate Gemini response (with user input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_gemini_response(input_prompt, file_data, prompt)
            except Exception as e:
                response = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
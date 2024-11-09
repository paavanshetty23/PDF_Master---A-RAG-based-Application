import streamlit as st
import PyPDF2
import io
from groq import Groq
import time

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def get_summary(text, client):
    """Generate summary using Groq API."""
    try:
        prompt = f"""Please provide a comprehensive summary of the following text. Focus on the main points and key takeaways:

        {text}

        Provide the summary in bullet points, highlighting the most important information."""

        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def answer_question(text, question, client):
    """Generate answer to user question using Groq API."""
    try:
        prompt = f"""Based on the following text, please answer the question accurately and concisely.

        Text: {text}

        Question: {question}

        Please provide a clear and direct answer based solely on the information provided in the text."""

        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="PDF Summarizer", page_icon="📄")

    # Application title and description
    st.title("📄 PDF Summarizer & Q&A")
    st.markdown("""
    Upload a PDF document to get a summary and ask questions about its content.
    This application uses advanced NLP to process your documents and provide insights.
    """)

    # Sidebar for API key input
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Groq API Key:", type="password")
        st.markdown("---")
        st.markdown("""
        ### How to use:
        1. Enter your Groq API key
        2. Upload a PDF file
        3. Get the summary
        4. Ask questions about the content
        """)

    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue.")
        return

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # File upload
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Create a container for the PDF text
            text_container = st.container()
            
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(uploaded_file)
            
            if pdf_text:
                # Show success message
                st.success("PDF processed successfully!")
                
                # Create tabs for different functionalities
                summary_tab, qa_tab, text_tab = st.tabs(["Summary", "Q&A", "Original Text"])
                
                with summary_tab:
                    if st.button("Generate Summary"):
                        with st.spinner("Generating summary..."):
                            summary = get_summary(pdf_text, client)
                            if summary:
                                st.markdown(summary)
                
                with qa_tab:
                    st.subheader("Ask Questions")
                    question = st.text_input("Enter your question about the document:")
                    if question and st.button("Get Answer"):
                        with st.spinner("Generating answer..."):
                            answer = answer_question(pdf_text, question, client)
                            if answer:
                                st.markdown("### Answer:")
                                st.markdown(answer)
                
                with text_tab:
                    st.subheader("Original Text")
                    st.text_area("Extracted text:", pdf_text, height=300)

if __name__ == "__main__":
    main()
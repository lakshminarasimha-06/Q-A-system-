import os
import logging
import fitz  
from PIL import Image
import pytesseract
import requests
import json
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
load_dotenv()
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


pytesseract.pytesseract.tesseract_cmd = r"C:\Users\hi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

PDF_DPI = 300  

API_KEY = os.getenv("API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """
    Convert PDF to list of PIL images
    """
    try:
        logger.info(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        logger.info(f"PDF Pages: {len(doc)}")
        
        images = []
        zoom = PDF_DPI/72  
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num, page in enumerate(doc, 1):
            logger.info(f"Processing page {page_num}")
            
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            logger.info(f"Converted page {page_num} to image")
        
        return images
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()

def perform_ocr(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Perform OCR on a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save OCR text
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        logger.info(f"Starting OCR for {pdf_path}")
        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path)
        
        # Perform OCR on each page
        all_text = ""
        for i, image in enumerate(images):
            # Perform OCR
            logger.info(f"Performing OCR on page {i+1}")
            page_text = pytesseract.image_to_string(image)
            
            # Add page separator
            all_text += f"\n\n--- Page {i+1} ---\n\n" + page_text
            
        # Save output to a text file if path is provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(all_text)
            logger.info(f"OCR output saved to {output_path}")
        
        return all_text
        
    except Exception as e:
        logger.error(f"Error during OCR process: {str(e)}")
        raise
def summarize_text(text: str, max_length: int = 500) -> str:
    """
    Summarizes text using the Dolphin-3.0 Mistral model on OpenRouter.

    Args:
        text (str): The text to summarize.
        max_length (int): Maximum length of the summary.

    Returns:
        str: The summarized text.
    """
    logger.info("Starting summarization with OpenRouter API")
    
    # If the text is too long, truncate it to a reasonable length for the API
    if len(text) > 12000:
        logger.warning(f"Text is too long ({len(text)} chars). Truncating to 12000 chars.")
        text = text[:12000] + "..."
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Document Summarization Task"
    }
    
    prompt = f"""Please provide a comprehensive summary of the following document. 
Focus on the main themes, key points, and important details. The summary should capture 
the essence of the document while being concise and well-structured.

DOCUMENT:
{text}"""
    
    payload = {
        "model": "cognitivecomputations/dolphin3.0-mistral-24b:free",  # Use the free Dolphin-3.0 model
        "messages": [
            {"role": "system", "content": "You are an AI assistant specialized in document summarization. Create concise, accurate, and comprehensive summaries that capture the key information from documents."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_length,
        "temperature": 0.5
    }

    try:
        logger.info("Sending request to OpenRouter API")
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.info("Successfully received summarization from OpenRouter API")
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return f"Error: API request failed: {str(e)}"
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        return f"Error: Summarization failed: {str(e)}"

def process_document(pdf_path: str, ocr_output_path: Optional[str] = None, 
                    summary_path: Optional[str] = None) -> Dict[str, str]:
    try:
        # Step 1 & 2: OCR processing
        logger.info(f"Starting document processing for {pdf_path}")
        extracted_text = perform_ocr(pdf_path, ocr_output_path)
        logger.info("OCR processing completed")
        
        # Step 3: Summarization
        logger.info("Starting summarization")
        summary = summarize_text(extracted_text)
        
        # Save summary if path is provided
        if summary_path:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            logger.info(f"Summary saved to {summary_path}")
        
        return {
            "extracted_text": extracted_text,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Document processing pipeline failed: {str(e)}")
        raise

def chunk_text_langchain(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter
    
    Args:
        text: Full document text
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    logger.info(f"Chunking text using LangChain's RecursiveCharacterTextSplitter with size={chunk_size}, overlap={chunk_overlap}")
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    
    logger.info(f"Created {len(chunks)} text chunks")
    return chunks

def setup_vector_store(text_chunks: List[str], persist_directory: str = "langchain_chroma_db"):
    """
    Set up vector store using LangChain and HuggingFace embeddings
    
    Args:
        text_chunks: List of text chunks
        persist_directory: Directory to persist the vector store
        
    Returns:
        LangChain vector store
    """
    logger.info("Setting up vector store with LangChain and HuggingFace embeddings")
    
    try:
        # Initialize the embedding model - using all-MiniLM-L6-v2 which is small and efficient
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create or load the Chroma vector store
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            logger.info(f"Loading existing vector store from {persist_directory}")
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            
            # Clear the existing collection to avoid data mixing
            vector_store.delete_collection()
            vector_store = Chroma.from_texts(
                texts=text_chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            logger.info(f"Creating new vector store at {persist_directory}")
            vector_store = Chroma.from_texts(
                texts=text_chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        
        logger.info("Vector store setup completed")
        return vector_store
    
    except Exception as e:
        logger.error(f"Error setting up vector store: {str(e)}")
        # Fallback to a simpler approach if there are issues
        logger.info("Falling back to basic embedding approach")
        return BasicVectorStore(text_chunks)

class BasicVectorStore:
    """
    A simple vector store implementation as a fallback when LangChain fails
    """
    def __init__(self, chunks):
        self.chunks = chunks
    
    def similarity_search(self, query, k=3):
        """
        A very basic search that looks for keyword matches
        """
        query_words = set(query.lower().split())
        chunk_scores = []
        
        for i, chunk in enumerate(self.chunks):
            # Calculate a simple relevance score based on word overlap
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            chunk_scores.append((i, overlap))
        
        # Sort by score and get top k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [self.chunks[idx] for idx, score in chunk_scores[:k]]
        
        # Format output to match LangChain's similarity_search return format
        return [{"page_content": chunk, "metadata": {}} for chunk in top_chunks]

def retrieve_relevant_chunks(query: str, vector_store, top_k: int = 3) -> List[str]:
    """
    Retrieve relevant chunks for a question using the vector store
    
    Args:
        query: User question
        vector_store: Vector store (LangChain or fallback)
        top_k: Number of chunks to retrieve
        
    Returns:
        List of relevant text chunks
    """
    logger.info(f"Retrieving top {top_k} chunks for query: {query}")
    
    try:
        # Try using the vector store's similarity search
        if hasattr(vector_store, 'similarity_search'):
            results = vector_store.similarity_search(query, k=top_k)
            chunks = [doc.page_content if hasattr(doc, 'page_content') else doc for doc in results]
        else:
            # For the fallback store
            results = vector_store.similarity_search(query, k=top_k)
            chunks = [doc["page_content"] for doc in results]
        
        logger.info(f"Retrieved {len(chunks)} relevant chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        # Fallback to returning random chunks if all else fails
        import random
        if hasattr(vector_store, 'chunks'):
            all_chunks = vector_store.chunks
        else:
            all_chunks = [doc.page_content for doc in vector_store.get()]
        
        random_chunks = random.sample(all_chunks, min(top_k, len(all_chunks)))
        logger.warning(f"Falling back to random selection of {len(random_chunks)} chunks")
        return random_chunks

def generate_answer(question: str, chunks: List[str]) -> str:
    """
    Generate an answer based on the question and relevant chunks
    
    Args:
        question: User question
        chunks: Relevant document chunks
        
    Returns:
        Generated answer
    """
    logger.info("Generating answer using OpenRouter API")
    
    # Combine chunks into context
    context = "\n\n".join(chunks)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Document Q&A Task"
    }
    
    prompt = f"""Answer the following question based ONLY on the provided document context.
If the information isn't available in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}"""
    
    payload = {
        "model": "cognitivecomputations/dolphin3.0-mistral-24b:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specializing in answering questions based on document contents. Use only the provided document context to answer questions."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        logger.info("Successfully generated answer from OpenRouter API")
        return answer
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return f"Error: API request failed: {str(e)}"
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        return f"Error: Answer generation failed: {str(e)}"

# Task 5: Streamlit Web Interface for Document Processing and Chatbot

def run_streamlit_app():
    """
    Run the Streamlit web application for document processing and Q&A
    """
    # Declare global variables at the beginning of the function
    global PDF_DPI
    
    # Configure the page
    try:
        st.set_page_config(page_title="Document Processing & Q&A System", layout="wide")
    except Exception:
        # This might happen if the script is run twice
        pass
    
    st.title("Intelligent Document Processing & Q&A System")
    st.write("Upload a PDF document, extract text with OCR, get a summary, and ask questions about the content.")
    
    # Sidebar for processing options
    with st.sidebar:
        st.header("Processing Options")
        dpi_option = st.slider("OCR DPI", min_value=100, max_value=600, value=PDF_DPI, step=50)
        chunk_size = st.slider("Text Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=200, step=50)
        top_k = st.slider("Retrieved Chunks", min_value=1, max_value=10, value=3, step=1)
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
        
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
        
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    
    # File upload and processing
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_pdf_path = "temp_upload.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Processing buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Update PDF_DPI variable
                    PDF_DPI = dpi_option
                    
                    # Process the document
                    ocr_output_path = "ocr_output.txt"
                    summary_path = "summary.txt"
                    
                    try:
                        results = process_document(
                            pdf_path=temp_pdf_path,
                            ocr_output_path=ocr_output_path,
                            summary_path=summary_path
                        )
                        
                        # Store results in session state
                        st.session_state.extracted_text = results["extracted_text"]
                        st.session_state.summary = results["summary"]
                        
                        # Chunk text using LangChain
                        chunks = chunk_text_langchain(
                            st.session_state.extracted_text,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        # Set up vector store
                        vector_store = setup_vector_store(chunks)
                        
                        # Store vector store in session state
                        st.session_state.vector_store = vector_store
                        st.session_state.pdf_processed = True
                        
                        st.success("Document processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
        
        # Display extracted text and summary if processed
        if st.session_state.pdf_processed:
            st.header("Document Processing Results")
            
            # Create tabs for displaying extracted text and summary
            tab1, tab2 = st.tabs(["Document Summary", "Extracted Text"])
            
            with tab1:
                st.subheader("Document Summary")
                st.write(st.session_state.summary)
                
                # Download button for summary
                st.download_button(
                    label="Download Summary",
                    data=st.session_state.summary,
                    file_name="document_summary.txt",
                    mime="text/plain"
                )
                
            with tab2:
                st.subheader("Extracted Text")
                st.text_area("OCR Result", st.session_state.extracted_text, height=400)
                
                # Download button for extracted text
                st.download_button(
                    label="Download Extracted Text",
                    data=st.session_state.extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
            
            # Q&A Interface
            st.header("Ask Questions About the Document")
            question = st.text_input("Enter your question:")
            
            if st.button("Ask"):
                if question.strip():
                    with st.spinner("Generating answer..."):
                        # Retrieve relevant chunks
                        chunks = retrieve_relevant_chunks(
                            question, 
                            st.session_state.vector_store,
                            top_k=top_k
                        )
                        
                        # Generate answer
                        answer = generate_answer(question, chunks)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({"question": question, "answer": answer})
                else:
                    st.warning("Please enter a question.")
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("Chat History")
                for i, chat in enumerate(st.session_state.chat_history):
                    st.write(f"**Question {i+1}**: {chat['question']}")
                    st.write(f"**Answer**: {chat['answer']}")
                    st.markdown("---")
                
                # Clear chat history button
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.experimental_rerun()

if __name__ == "__main__":
    # Import system modules
    import sys
    
    # Detect if we're running in Streamlit
    in_streamlit = 'streamlit' in sys.modules
    
    if in_streamlit:
        # If running with streamlit, just call the app function
        run_streamlit_app()
    else:
        # CLI mode
        # Replace with the path to your PDF
        pdf_path = "The_Gift_of_the_Magi.pdf"
        ocr_output_path = "ocr_output.txt"
        summary_path = "summary_Magi.txt"
        
        try:
            # Process the document
            results = process_document(
                pdf_path=pdf_path,
                ocr_output_path=ocr_output_path,
                summary_path=summary_path
            )
            
            print("\n===== DOCUMENT PROCESSING COMPLETED =====")
            print(f"OCR Text saved to: {ocr_output_path}")
            print(f"Summary saved to: {summary_path}")
            print("\nSummary:")
            print(results["summary"])
            
            # Set up the RAG system
            print("\n===== SETTING UP RAG SYSTEM =====")
            
            # Chunk text using LangChain
            chunks = chunk_text_langchain(results["extracted_text"])
            print(f"Text split into {len(chunks)} chunks")
            
            # Set up vector store
            vector_store = setup_vector_store(chunks)
            print("Vector store setup completed")
            
            print("\n===== RAG SYSTEM READY =====")
            print("You can now ask questions. Type 'exit' to quit.")
            
            # Simple CLI Q&A loop
            while True:
                question = input("\nYour question: ")
                if question.lower() == 'exit':
                    break
                
                try:
                    # Retrieve relevant chunks
                    chunks = retrieve_relevant_chunks(question, vector_store)
                    
                    # Generate answer
                    answer = generate_answer(question, chunks)
                    
                    print("\nAnswer:")
                    print(answer)
                except Exception as e:
                    print(f"Error processing question: {str(e)}")
                    
        except Exception as e:
            print(f"Error: {str(e)}")
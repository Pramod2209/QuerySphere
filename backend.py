import fitz  # PyMuPDF
import faiss
import numpy as np
import json
import io
import email
from email.policy import default
import docx # For .docx files
from groq import Groq
import requests
from io import BytesIO

def get_file_from_blob_url(blob_url: str):
    """Downloads file from blob URL and returns file-like object"""
    try:
        response = requests.get(blob_url)
        response.raise_for_status()
        return BytesIO(response.content)
    except Exception as e:
        print(f"ERROR: Failed to download file from blob URL: {e}")
        return None

def extract_text_from_document(file_or_blob_url) -> list[dict] | None:
    """Extracts text chunks from a file or blob URL (PDF, DOCX, or EML)."""
    text_chunks = []
    
    try:
        # Handle blob URL case
        if isinstance(file_or_blob_url, str) and file_or_blob_url.startswith(('http://', 'https://')):
            file_bytes = get_file_from_blob_url(file_or_blob_url)
            if not file_bytes:
                return None
            file_name = file_or_blob_url.split('/')[-1].split('?')[0]  # Remove query params
        else:
            # Handle file upload case
            file_name = file_or_blob_url.name
            file_bytes = BytesIO(file_or_blob_url.getvalue())
        
        if file_name.lower().endswith('.pdf'):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks")
                for block in blocks:
                    text = block[4].replace('\n', ' ').strip()
                    if len(text) > 40:
                        text_chunks.append({
                            "text": text, 
                            "page": page_num + 1,
                            "source": f"PDF page {page_num + 1}"
                        })
        
        elif file_name.lower().endswith('.docx'):
            doc = docx.Document(file_bytes)
            for para_num, para in enumerate(doc.paragraphs):
                text = para.text.strip()
                if len(text) > 40:
                    text_chunks.append({
                        "text": text,
                        "page": para_num + 1,
                        "source": f"Section {para_num + 1}"
                    })

        elif file_name.lower().endswith('.eml'):
            msg = email.message_from_bytes(file_bytes.getvalue(), policy=default)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if content_type == 'text/plain' and 'attachment' not in content_disposition:
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                if msg.get_content_type() == 'text/plain':
                    body = msg.get_payload(decode=True).decode()
            
            for para_num, para in enumerate(body.split('\n\n')):
                text = para.replace('\n', ' ').strip()
                if len(text) > 40:
                    text_chunks.append({
                        "text": text,
                        "page": para_num + 1,
                        "source": f"Paragraph {para_num + 1}"
                    })

        return text_chunks if text_chunks else None

    except Exception as e:
        print(f"ERROR: Failed to process file {file_name}.\nDetails: {e}")
        return None

def create_faiss_index(text_chunks: list[str], embedding_model):
    """Creates a FAISS index using a loaded SentenceTransformer model."""
    try:
        embeddings = embedding_model.encode(text_chunks, show_progress_bar=False)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index
    except Exception as e:
        print(f"ERROR: Failed to create local embeddings.\nDetails: {e}")
        return None

def synthesize_answer_with_groq(client, query: str, retrieved_clauses: list[dict]):
    """Generates a synthesized answer using Llama 3 on Groq."""
    context = "\n\n".join([f"Source: {c['source']}\nText: {c['text']}" for c in retrieved_clauses])
    
    system_prompt = """
    You are a helpful assistant designed to answer questions about a document.
    Your response MUST be a valid JSON object with these keys: 
    - "relevant_clause": The most relevant text from the context
    - "explanation": How the clause answers the query
    - "source_reference": Where the clause was found
    If the answer cannot be found, state that in "explanation".
    """
    user_prompt = f"Based ONLY on this CONTEXT, answer the QUERY.\n\nCONTEXT:\n{context}\n\nQUERY: {query}"
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"ERROR: Failed to generate answer with Groq/Llama 3.\nDetails: {e}")
        return None

def process_query(api_key: str, file_or_url, query: str, embedding_model) -> dict:
    """Main processing pipeline to handle a user query against a document."""
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return {"error": f"Failed to initialize Groq client. Check your API key. Details: {e}"}

    document_chunks = extract_text_from_document(file_or_url)
    if not document_chunks:
        return {"error": "Could not extract text. The file might be empty, image-based, or in an unsupported format."}
    
    document_texts = [chunk['text'] for chunk in document_chunks]

    index = create_faiss_index(document_texts, embedding_model)
    if not index:
        return {"error": "Failed to create document embeddings."}

    try:
        query_embedding = embedding_model.encode([query])
        k = 3
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
        retrieved_clauses = [document_chunks[i] for i in indices[0]]
    except Exception as e:
        return {"error": f"Failed during search. Details: {e}"}

    synthesized_answer = synthesize_answer_with_groq(client, query, retrieved_clauses)
    
    if not synthesized_answer:
        return {
            "explanation": "Model failed to generate response. Showing most relevant clause.",
            "relevant_clause": retrieved_clauses[0]['text'],
            "source_reference": retrieved_clauses[0]['source']
        }
    
    try:
        return json.loads(synthesized_answer)
    except json.JSONDecodeError:
        return {
            "explanation": "Invalid JSON response. Showing most relevant clause.",
            "relevant_clause": retrieved_clauses[0]['text'],
            "source_reference": retrieved_clauses[0]['source']
        }
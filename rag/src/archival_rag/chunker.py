"""
RAG chunking pipeline for German Berufsausbildung ordinance PDFs.
Hybrid strategy: legal §-paragraphs are chunked as prose sections,
Anlage/Ausbildungsrahmenplan tables are chunked row-by-row.
"""

import os
import re
import pymupdf as fitz 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Regex patterns used to detect structure in the extracted text
PARAGRAPH_PATTERN = re.compile(r"(?m)^\s*§\s*\d+[a-z]?\b.*$")
ANLAGE_PATTERN = re.compile(r"(?im)^\s*Anlage\b")
TABLE_ROW_PATTERN = re.compile(r"(?m)^\s*(\d{1,2})\s+(.+)")
YEAR_PATTERN = re.compile(r"\b(19[6-9]\d|20[0-2]\d)\b")


# Reads a PDF page by page and returns the full concatenated text.
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


# Cleans OCR whitespace noise (multiple spaces/newlines) without touching content.
def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Pulls a rough document title and issue year from the text, used as chunk metadata.
def extract_doc_metadata(text, filename):
    title_match = re.search(r"(?im)^\s*Verordnung.*?\n(.*)", text)
    title = title_match.group(1).strip() if title_match else filename
    year_match = YEAR_PATTERN.search(filename) or YEAR_PATTERN.search(text)
    year = year_match.group(0) if year_match else "unknown"
    return {"title": title, "year": year, "source": filename}


# Splits the document into (a) prose part before "Anlage" and (b) table part from "Anlage" onward.
def split_prose_and_table(text):
    match = ANLAGE_PATTERN.search(text)
    if match:
        return text[:match.start()], text[match.start():]
    return text, ""


# Splits the prose part into one chunk per § paragraph using regex boundaries.
def chunk_paragraphs(prose_text, base_meta):
    positions = [m.start() for m in PARAGRAPH_PATTERN.finditer(prose_text)]
    if not positions:
        return chunk_fallback(prose_text, base_meta)  # no § found, use fallback splitter

    positions.append(len(prose_text))
    chunks = []
    for i in range(len(positions) - 1):
        section_text = prose_text[positions[i]:positions[i + 1]].strip()
        if not section_text:
            continue
        para_id = PARAGRAPH_PATTERN.match(section_text).group(0).strip()
        meta = {**base_meta, "section": para_id, "chunk_type": "paragraph"}
        # split further only if the § section itself is long
        chunks.extend(chunk_fallback(section_text, meta) if len(section_text) > 1200
                       else [Document(page_content=section_text, metadata=meta)])
    return chunks


# Splits the Anlage/table part row-by-row so each catalog entry becomes one chunk.
def chunk_table_rows(table_text, base_meta):
    rows = list(TABLE_ROW_PATTERN.finditer(table_text))
    if not rows:
        return chunk_fallback(table_text, base_meta)

    header = table_text[:rows[0].start()].strip()  # table title / column captions as shared context
    chunks = []
    for i, row_match in enumerate(rows):
        end = rows[i + 1].start() if i + 1 < len(rows) else len(table_text)
        row_text = table_text[row_match.start():end].strip()
        content = f"{header}\n\n{row_text}" if header else row_text
        meta = {**base_meta, "section": f"Anlage row {row_match.group(1)}", "chunk_type": "table_row"}
        chunks.append(Document(page_content=content, metadata=meta))
    return chunks


# Generic fallback splitter for text that doesn't match the expected § or table patterns.
def chunk_fallback(text, base_meta):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return [Document(page_content=c, metadata={**base_meta, "chunk_type": base_meta.get("chunk_type", "fallback")})
            for c in splitter.split_text(text)]


# Runs the full extraction + chunking pipeline for a single PDF file.
def process_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    raw_text = clean_text(extract_text_from_pdf(pdf_path))
    base_meta = extract_doc_metadata(raw_text, filename)

    prose_text, table_text = split_prose_and_table(raw_text)
    chunks = chunk_paragraphs(prose_text, base_meta)
    if table_text:
        chunks.extend(chunk_table_rows(table_text, base_meta))
    return chunks


# Iterates over every PDF in a folder and collects all chunks into one list.
def process_folder(folder_path):
    all_chunks = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(folder_path, fname)
            all_chunks.extend(process_pdf(path))
    return all_chunks


# Creates the Qdrant collection if it doesn't already exist, sized to match the embedding model.
def ensure_collection(client, collection_name, vector_size):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


# Embeds all chunks with a HuggingFace model and upserts them into your Qdrant Cloud collection.
def build_vectorstore(chunks, qdrant_url, qdrant_api_key, collection_name="berufsausbildung",
                       model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    vector_size = len(embeddings.embed_query("dimension probe"))
    ensure_collection(client, collection_name, vector_size)

    vectorstore = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    vectorstore.add_documents(chunks)
    return vectorstore


if __name__ == "__main__":
    folder_path = "/Users/faiqimran6/projects/lab-sommer-2026/pdfs"  # set your local folder path here if you are taking pdfs from local machine
    QDRANT_URL = "https://a45480a6-6c45-4049-9080-78a5dd425dd8.eu-central-1-0.aws.cloud.qdrant.io:6333"
    QDRANT_API_KEY = qdrant_api_key  # put your own key (in my case it is api key for cloud instance)

    chunks = process_folder(folder_path)
    print(f"Created {len(chunks)} chunks from folder: {folder_path}")

    vectorstore = build_vectorstore(chunks, qdrant_url=QDRANT_URL, qdrant_api_key=QDRANT_API_KEY)
    print("Chunks embedded and upserted to Qdrant Cloud.")

    # as an example, this query is taken 
    results = vectorstore.similarity_search("Wie lange dauert die Ausbildung?", k=3)
    for r in results:
        print(r.metadata, "->", r.page_content[:150])
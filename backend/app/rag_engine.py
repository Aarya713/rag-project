import os
import numpy as np
from typing import List, Tuple, Optional
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq

from .models import AnswerResponse, SourceInfo

# Load .env at module level – safe to call multiple times
load_dotenv()

class RAGEngine:
    def __init__(self, device: int = -1):
        self.device = device
        # All heavy components are None initially
        self._embeddings = None
        self._docling_converter = None
        self.vectorstore = None
        self.bm25 = None
        self.chunks = []
        self.llm = None
        self.history = []
        print("📦 RAG engine initialized. Heavy components will load only when needed.")

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embeddings

    def _get_converter(self):
        if self._docling_converter is None:
            from docling.document_converter import DocumentConverter
            self._docling_converter = DocumentConverter()
        return self._docling_converter

    def _load_llm(self):
        if self.llm is not None:
            return
        # Reload .env just to be safe
        load_dotenv()
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment. Please add it to .env file.")
        
        try:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",  # Active model
                temperature=0.3,
                max_tokens=512,
                api_key=api_key
            )
            print("✅ Groq LLM loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load Groq: {e}")
            raise

    def parse_document(self, pdf_path: str) -> Tuple[str, List[Document]]:
        converter = self._get_converter()
        result = converter.convert(pdf_path)
        doc = result.document
        full_text = doc.export_to_markdown()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        page_chunks = []
        if hasattr(doc, "pages"):
            for page_num, page in doc.pages.items():
                page_text = page.text if hasattr(page, "text") else ""
                if page_text.strip():
                    page_raw = text_splitter.split_text(page_text)
                    for chunk_text in page_raw:
                        page_chunks.append(Document(
                            page_content=chunk_text,
                            metadata={"source": os.path.basename(pdf_path), "page": page_num}
                        ))
        if not page_chunks:
            raw_chunks = text_splitter.split_text(full_text)
            for chunk_text in raw_chunks:
                page_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={"source": os.path.basename(pdf_path)}
                ))

        # Memory cleanup
        try:
            if hasattr(result.input, '_backend'):
                result.input._backend.unload()
        except:
            pass

        return full_text, page_chunks

    def process_pdfs(self, file_paths: List[str]) -> Tuple[int, int]:
        all_chunks = []
        for path in file_paths:
            _, chunks = self.parse_document(path)
            all_chunks.extend(chunks)

        self.chunks = all_chunks
        if not self.chunks:
            raise ValueError("No text chunks extracted from the PDFs.")

        embeddings = self._get_embeddings()
        self.vectorstore = FAISS.from_documents(self.chunks, embeddings)

        tokenized_chunks = [chunk.page_content.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        return len(self.chunks), len(file_paths)

    def _hybrid_search(self, query: str, k: int = 4) -> List[Document]:
        vector_docs = self.vectorstore.similarity_search(query, k=k)
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-k:][::-1]
        bm25_docs = [self.chunks[i] for i in top_indices if bm25_scores[i] > 0]

        seen = set()
        combined = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                combined.append(doc)
                seen.add(doc.page_content)

        return combined[:k]

    def ask(self, question: str) -> AnswerResponse:
        if self.vectorstore is None:
            raise ValueError("No PDFs processed yet. Please upload first.")

        self._load_llm()

        docs = self._hybrid_search(question, k=4)
        context = "\n\n---\n\n".join([d.page_content for d in docs])

        history_text = ""
        if self.history:
            last_entries = self.history[-3:]
            history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in last_entries])

        prompt = f"""Previous conversation:
{history_text}

Context:
{context}

Question: {question}
Answer (strictly from context, say "I could not find this information in the document" if not present):"""
        try:
            raw = self.llm.invoke(prompt)
            if hasattr(raw, 'content'):
                answer = raw.content.strip()
            else:
                answer = str(raw).strip()
        except Exception as e:
            raise Exception(f"Error calling Groq API: {e}")

        self.history.append((question, answer))

        # ============================================================
        # 🔽 TOGGLE SOURCES DISPLAY
        # Set this to True to show sources, False to hide them.
        # ============================================================
        SHOW_SOURCES = False  
        # ============================================================

        if SHOW_SOURCES:
            sources = []
            for d in docs:
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page")
                sources.append(SourceInfo(source=src, page=page))
        else:
            sources = []

        return AnswerResponse(answer=answer, sources=sources)
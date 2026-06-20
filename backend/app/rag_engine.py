import os
import numpy as np
from typing import List, Tuple, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq
from docling.document_converter import DocumentConverter

from .models import AnswerResponse, SourceInfo

class RAGEngine:
    def __init__(self, device: int = -1):
        self.device = device
        # LAZY LOAD: Don't load embeddings immediately
        self._embeddings = None
        self._docling_converter = None
        self.vectorstore = None
        self.bm25 = None
        self.chunks = []
        self.llm = None
        self.history = []
        print("📦 RAG engine initialized (components will load on demand).")

    def _get_embeddings(self):
        """Load HuggingFace embeddings only when needed."""
        if self._embeddings is None:
            print("🔄 Loading embedding model...")
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✅ Embedding model loaded.")
        return self._embeddings

    def _get_converter(self):
        """Load DocumentConverter only when needed."""
        if self._docling_converter is None:
            print("🔄 Loading document converter...")
            self._docling_converter = DocumentConverter()
            print("✅ Document converter loaded.")
        return self._docling_converter

    def _load_llm(self):
        """Load Groq LLM lazily – completely free!"""
        if self.llm is not None:
            return
        print("🤖 Connecting to Groq API (Free tier)...")
        try:
            self.llm = ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=512
            )
            print("✅ Groq connected successfully!")
        except Exception as e:
            print(f"❌ Failed to connect to Groq: {e}")
            raise

    def parse_document(self, pdf_path: str) -> Tuple[str, List[Document]]:
        # Use lazy-loaded converter
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

        return full_text, page_chunks

    def process_pdfs(self, file_paths: List[str]) -> Tuple[int, int]:
        all_chunks = []
        for path in file_paths:
            _, chunks = self.parse_document(path)
            all_chunks.extend(chunks)

        self.chunks = all_chunks

        if not self.chunks:
            raise ValueError("No text chunks extracted from the PDFs.")

        # Now we actually need embeddings, so load them
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

        # Lazy-load the LLM only when the first question is asked
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
        raw = self.llm.invoke(prompt)
        if hasattr(raw, 'content'):
            answer = raw.content.strip()
        else:
            answer = str(raw).strip()

        self.history.append((question, answer))

        sources = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            sources.append(SourceInfo(source=src, page=page))

        return AnswerResponse(answer=answer, sources=sources)
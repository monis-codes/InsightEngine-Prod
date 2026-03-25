import os
import hashlib
import math
from io import BytesIO
from typing import Any, List

from pypdf import PdfReader
import google.generativeai as genai


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Environment variable '{var_name}' is required.")
    return value


def _extract_text_from_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    if len(reader.pages) == 0:
        raise ValueError("PDF contains zero pages")
    text_parts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    text = " ".join(text_parts)
    text = " ".join(text.split())
    if not text:
        raise ValueError("No extractable text found in the PDF")
    return text


def _normalize_vector(vector: List[float]) -> List[float]:
    # For reduced-dimensional embeddings, normalize for stable cosine behavior.
    magnitude = math.sqrt(sum((v * v) for v in vector))
    if magnitude == 0:
        return vector
    return [v / magnitude for v in vector]


def _embedding_config() -> tuple[str, int]:
    model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001").strip()
    if not model:
        model = "models/gemini-embedding-001"
    if not model.startswith("models/"):
        model = f"models/{model}"

    raw_dim = (os.getenv("GOOGLE_EMBEDDING_DIMENSION", "768") or "768").strip()
    try:
        dimension = int(raw_dim)
    except ValueError:
        dimension = 768
    return model, dimension


def _chunk_text(text: str, max_chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    if not text or not text.strip():
        return []
    sentences = text.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() + "." for s in sentences if s.strip()]

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chunk_size:
            if current:
                chunks.append(current.strip())
                words = current.split()
                overlap_text = " ".join(words[-overlap // 10:]) if len(words) > overlap // 10 else ""
                current = (overlap_text + " " + sentence).strip() if overlap_text else sentence
            else:
                chunks.append(sentence[:max_chunk_size])
                current = sentence[max_chunk_size:]
        else:
            current = (current + " " + sentence).strip() if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c.strip()) > 50]


def process_document(index: Any, document_content: bytes, namespace: str) -> bool:
    if not index:
        raise ValueError("Index cannot be None")
    if not document_content:
        raise ValueError("Document content cannot be empty")
    if not namespace or not namespace.strip():
        raise ValueError("Namespace cannot be empty")

    # Configure Generative AI
    genai.configure(api_key=_require_env("GOOGLE_API_KEY"))

    # Only PDF is supported in this minimal setup
    if not document_content.startswith(b"%PDF"):
        raise ValueError("Unsupported file type. Only PDF is supported in this setup.")

    full_text = _extract_text_from_pdf(document_content)
    chunks = _chunk_text(full_text, max_chunk_size=1500)
    if not chunks:
        raise ValueError("Failed to create any text chunks from the document")

    embedding_model, embedding_dimension = _embedding_config()
    result = genai.embed_content(
        model=embedding_model,
        content=chunks,
        task_type="RETRIEVAL_DOCUMENT",
        title="Document Chunks",
        output_dimensionality=embedding_dimension
    )
    embeddings: List[List[float]] = result["embedding"] if isinstance(result, dict) else result.embedding

    vectors = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        normalized_vector = _normalize_vector(vector)
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
        vector_id = f"{namespace}-{chunk_hash}-{i}"
        vectors.append({
            "id": vector_id,
            "values": normalized_vector,
            "metadata": {
                "text": chunk,
                "chunk_index": i,
                "namespace": namespace,
                "char_count": len(chunk),
                "file_type": "pdf"
            }
        })

    # Upsert in batches
    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
    return True


def get_answer(index: Any, question: str, namespace: str) -> str:
    if not index:
        raise ValueError("Index cannot be None")
    if not question or not question.strip():
        return "Please enter a valid question."
    if not namespace or not namespace.strip():
        return "No active document. Please upload and process a document first."

    genai.configure(api_key=_require_env("GOOGLE_API_KEY"))

    embedding_model, embedding_dimension = _embedding_config()
    search_embedding = genai.embed_content(
        model=embedding_model,
        content=question,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=embedding_dimension
    )["embedding"]
    search_embedding = _normalize_vector(search_embedding)

    matches = index.query(
        namespace=namespace,
        vector=search_embedding,
        top_k=8,
        include_metadata=True
    ).get("matches", [])

    if not matches:
        return "I couldn't find relevant information in the processed document."

    context_parts = [m["metadata"].get("text", "") for m in matches]
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "Answer the question using ONLY the provided context. Be precise and concise.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0.0, "max_output_tokens": 1024})
    return (getattr(response, "text", "") or "No answer generated.").strip()




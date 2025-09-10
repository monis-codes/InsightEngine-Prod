#💡 Insight Engine: A Resource-Conscious RAG Demo
This project is a lightweight, proof-of-concept version of a Retrieval-Augmented Generation (RAG) pipeline. It is specifically designed to be computationally efficient and easily deployable for demonstration purposes, making it an ideal tool for showcasing the core RAG architecture to peers and recruiters.

While this public-facing demo is streamlined, it represents a foundational component of a much more sophisticated system. The full, optimized version of this project is too computationally intensive for free public deployment due to its GPU-heavy requirements for running advanced neural network models.

For a detailed look at the complete, optimized codebase, please refer to the [rxHackathon] repository. This demo itself is open source, and we welcome contributions to the [rag-Deploy] repository.

##How This Lite Version Works
This demo encapsulates a complete, end-to-end RAG workflow, tailored for efficiency and clarity. It performs three main tasks:

Document Processing: The app accepts a PDF file as input. It extracts all the text and then uses a custom "smart chunking" algorithm. This method splits the text into meaningful segments while ensuring sentences are not broken and a strategic overlap is maintained. This approach is more effective for maintaining context than simple fixed-size splitting.

Smart Chunking: This is a crucial feature that ensures text chunks are semantically coherent. The algorithm processes text sentence by sentence, intelligently grouping them into chunks of a predefined size while preserving a small overlap between consecutive chunks. This overlap helps the RAG model maintain conversational flow and context across chunk boundaries.

Embedding & Vector Indexing: Once the text is chunked, each chunk is converted into a high-dimensional vector using Google's powerful models/text-embedding-004 embedding model. These embeddings are then stored in a Pinecone vector database. Pinecone acts as our efficient, scalable knowledge base, allowing for ultra-fast semantic searches.

Retrieval & Generation: When a user enters a question, the question itself is also converted into an embedding. The Pinecone index is then queried to find the top 8 most semantically similar text chunks from the processed document. These chunks are then used as the context for a large language model (gemini-2.5-flash), which generates a precise and concise answer based only on the provided context.

##Key Architectural Choices for the Lite Version
This version was built with specific design constraints to make it lightweight and deployable without expensive hardware:

Google's Generative AI Models: Instead of using computationally heavy, self-hosted models, this project leverages Google's efficient API for both embeddings (text-embedding-004) and generation (gemini-2.5-flash). This offloads the heavy lifting to a robust cloud service.

Pinecone Vector Database: A managed vector database like Pinecone is used to handle the search and retrieval, eliminating the need to run and manage an in-memory or local vector database.

Simple Chunks: The smart chunking algorithm is a lightweight and effective way to prepare text, removing the need for more complex chunking methods that would be part of the full project (like HyDE-based generation).

##Getting Started
Clone the repository: git clone https://github.com/monis-codes/rag-Deploy.git

Set up environment variables: You will need a GOOGLE_API_KEY and a Pinecone API key and environment.

Install dependencies: pip install -r requirements.txt

Run the app: streamlit run app.py

Note: This project is an open-source endeavor. Contributions are welcome to the main rag-Deploy repository to help us build a more robust and feature-rich RAG pipeline.
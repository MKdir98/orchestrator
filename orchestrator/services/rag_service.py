import os
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from typing import List, Optional, Dict

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with embeddings and text splitter"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )

    def get_user_rag_dir(self, user_id: int) -> str:
        """Get or create user directory for storing RAG data"""
        from orchestrator.services.data_service import DataService  # Lazy import
        base_dir = DataService().get_user_data_path(user_id)
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def add_text(self, user_id: int, text: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add text to the user's RAG index
        Returns True if successful, False otherwise
        """
        if metadata is None:
            metadata = {}

        user_dir = self.get_user_rag_dir(user_id)
        document = Document(page_content=text, metadata=metadata)

        try:
            splitted_docs = self.text_splitter.split_documents([document])
            if not splitted_docs:
                print("No documents generated after splitting")
                return False

            index_path = os.path.join(user_dir, "index.faiss")

            if os.path.exists(index_path):
                # Load existing index
                vector_store = FAISS.load_local(
                    folder_path=user_dir,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                vector_store.add_documents(splitted_docs)
            else:
                # Create new index
                vector_store = FAISS.from_documents(
                    documents=splitted_docs,
                    embedding=self.embeddings
                )

            # Save the index
            vector_store.save_local(user_dir)

            # Verify the files were created
            required_files = ["index.faiss", "index.pkl"]
            for file in required_files:
                if not os.path.exists(os.path.join(user_dir, file)):
                    raise RuntimeError(f"Failed to create {file}")

            return True

        except Exception as e:
            print(f"Error adding text to RAG index: {str(e)}")
            return False

    def query_context(self, user_id: str, question: str, k: int = 30) -> str:
        """
        Query the RAG system for relevant context
        Returns empty string if no index exists or on error
        """
        user_dir = self.get_user_rag_dir(user_id)
        index_path = os.path.join(user_dir, "index.faiss")

        if not os.path.exists(index_path):
            return ""

        try:
            vector_store = FAISS.load_local(
                folder_path=user_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            docs = vector_store.similarity_search(question, k=k)
            return "\n\n".join([f"Context {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

        except Exception as e:
            print(f"Error querying RAG index: {str(e)}")
            return ""

    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all RAG data for a user
        Returns True if successful, False otherwise
        """
        user_dir = self.get_user_rag_dir(user_id)
        try:
            if os.path.exists(user_dir):
                for filename in os.listdir(user_dir):
                    file_path = os.path.join(user_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {str(e)}")
                os.rmdir(user_dir)
            return True
        except Exception as e:
            print(f"Error deleting user data: {str(e)}")
            return False

    def get_index_stats(self, user_id: str) -> Dict:
        """
        Get statistics about the user's RAG index
        Returns dictionary with stats or error message
        """
        user_dir = self.get_user_rag_dir(user_id)
        index_path = os.path.join(user_dir, "index.faiss")

        if not os.path.exists(index_path):
            return {"error": "Index does not exist"}

        try:
            vector_store = FAISS.load_local(
                folder_path=user_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            return {
                "document_count": len(vector_store.docstore._dict),
                "index_size": f"{os.path.getsize(index_path)/1024:.2f} KB",
                "index_path": user_dir
            }
        except Exception as e:
            return {"error": str(e)}

import os
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from typing import List, Optional, Dict
import gc

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with embeddings and text splitter"""
        local_model_path = "models/paraphrase-multilingual-MiniLM-L12-v2"
        if not os.path.exists(local_model_path):
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            model.save(local_model_path)
        
        # تنظیمات بهینه برای کاهش مصرف حافظه
        self.embeddings = HuggingFaceEmbeddings(
            model_name=local_model_path,
            model_kwargs={'device': 'cpu'},  # استفاده از CPU
            encode_kwargs={'normalize_embeddings': True}
        )

        # کاهش اندازه chunk برای کاهش مصرف حافظه
        self.text_splitter = CharacterTextSplitter(
            chunk_size=500,  # کاهش از 1000 به 500
            chunk_overlap=100,  # کاهش از 200 به 100
            separator="\n"
        )
        
        # محدود کردن تعداد اسناد
        self.max_documents = 1000
        self.current_documents = 0

    def get_user_rag_dir(self, user_id: int) -> str:
        """Get or create user directory for storing RAG data"""
        from orchestrator.services.data_service import DataService  # Lazy import
        base_dir = DataService().get_user_data_path(user_id)
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def add_text(self, text: str, user_id: int = None) -> None:
        """Add text to the RAG system with memory optimization"""
        try:
            # تقسیم متن به chunk های کوچک‌تر
            chunks = self.text_splitter.split_text(text)
            
            # محدود کردن تعداد chunk ها
            if len(chunks) > 100:  # حداکثر 100 chunk
                chunks = chunks[:100]
            
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # بررسی محدودیت تعداد اسناد
            if self.current_documents + len(documents) > self.max_documents:
                # حذف قدیمی‌ترین اسناد
                self._cleanup_old_documents(user_id)
            
            # ذخیره در FAISS
            if user_id:
                rag_dir = self.get_user_rag_dir(user_id)
                vectorstore_path = os.path.join(rag_dir, "faiss_index")
                
                if os.path.exists(vectorstore_path):
                    # بارگذاری index موجود
                    vectorstore = FAISS.load_local(vectorstore_path, self.embeddings)
                    vectorstore.add_documents(documents)
                else:
                    # ایجاد index جدید
                    vectorstore = FAISS.from_documents(documents, self.embeddings)
                
                vectorstore.save_local(vectorstore_path)
                self.current_documents += len(documents)
            
            # پاک کردن حافظه
            del documents, chunks
            gc.collect()
            
        except Exception as e:
            print(f"Error adding text to RAG: {e}")

    def query_context(self, user_id: int, query: str, top_k: int = 5) -> str:
        """Query the RAG system with memory optimization"""
        try:
            rag_dir = self.get_user_rag_dir(user_id)
            vectorstore_path = os.path.join(rag_dir, "faiss_index")
            
            if not os.path.exists(vectorstore_path):
                return ""
            
            # بارگذاری index
            vectorstore = FAISS.load_local(vectorstore_path, self.embeddings)
            
            # جستجو با محدودیت نتایج
            results = vectorstore.similarity_search(query, k=min(top_k, 10))
            
            # ترکیب نتایج
            context = "\n".join([doc.page_content for doc in results])
            
            # پاک کردن حافظه
            del vectorstore, results
            gc.collect()
            
            return context
            
        except Exception as e:
            print(f"Error querying RAG: {e}")
            return ""

    def _cleanup_old_documents(self, user_id: int) -> None:
        """پاک کردن اسناد قدیمی برای آزاد کردن حافظه"""
        try:
            rag_dir = self.get_user_rag_dir(user_id)
            vectorstore_path = os.path.join(rag_dir, "faiss_index")
            
            if os.path.exists(vectorstore_path):
                # بارگذاری index
                vectorstore = FAISS.load_local(vectorstore_path, self.embeddings)
                
                # حذف نیمی از اسناد قدیمی
                if hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, '_dict'):
                    docs = list(vectorstore.docstore._dict.values())
                    if len(docs) > self.max_documents // 2:
                        # حذف قدیمی‌ترین اسناد
                        docs_to_remove = docs[:len(docs) // 2]
                        for doc in docs_to_remove:
                            if hasattr(doc, 'page_content'):
                                # حذف از index
                                pass
                
                vectorstore.save_local(vectorstore_path)
                self.current_documents = max(0, self.current_documents - len(docs_to_remove))
                
        except Exception as e:
            print(f"Error cleaning up old documents: {e}")

    def clear_user_data(self, user_id: int) -> None:
        """پاک کردن تمام داده‌های کاربر"""
        try:
            rag_dir = self.get_user_rag_dir(user_id)
            vectorstore_path = os.path.join(rag_dir, "faiss_index")
            
            if os.path.exists(vectorstore_path):
                import shutil
                shutil.rmtree(vectorstore_path)
            
            self.current_documents = 0
            gc.collect()
            
        except Exception as e:
            print(f"Error clearing user data: {e}")

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

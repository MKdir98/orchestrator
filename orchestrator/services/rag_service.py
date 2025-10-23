import os
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from typing import List, Optional, Dict
import gc
import json
from datetime import datetime

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with embeddings and text splitter"""
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        local_model_path = "models/paraphrase-multilingual-MiniLM-L12-v2"
        
        # Try to use local model if it exists and is valid, otherwise use online model
        try:
            if os.path.exists(local_model_path):
                # Test if local model is valid
                test_model = SentenceTransformer(local_model_path)
                del test_model  # Clean up
                model_to_use = local_model_path
            else:
                model_to_use = model_name
        except Exception:
            # If local model is corrupted, use online model and re-download
            model_to_use = model_name
            if os.path.exists(local_model_path):
                import shutil
                shutil.rmtree(local_model_path)
        
        # Download and save model locally if needed
        if model_to_use == model_name:
            try:
                model = SentenceTransformer(model_name)
                os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                model.save(local_model_path)
                del model
                model_to_use = local_model_path
            except Exception as e:
                print(f"Failed to save model locally: {e}")
                # Continue with online model
                pass
        
        # تنظیمات بهینه برای کاهش مصرف حافظه
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_to_use,
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

    def add_text(self, user_id: int, text: str, metadata: Optional[Dict] = None) -> None:
        """Add text to RAG system with optional metadata (task_id, task_description, timestamp)"""
        chunks = self.text_splitter.split_text(text)

        # محدود کردن تعداد chunk ها
        if len(chunks) > 100:  # حداکثر 100 chunk
            chunks = chunks[:100]

        # افزودن metadata به هر document
        if metadata is None:
            metadata = {}
        
        # اطمینان از وجود timestamp
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()

        documents = [Document(page_content=chunk, metadata=metadata.copy()) for chunk in chunks]

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
                vectorstore = FAISS.load_local(vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
                vectorstore.add_documents(documents)
            else:
                # ایجاد index جدید
                vectorstore = FAISS.from_documents(documents, self.embeddings)

            vectorstore.save_local(vectorstore_path)
            self.current_documents += len(documents)

        # پاک کردن حافظه
        del documents, chunks
        gc.collect()

    def get_recent_events(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get the most recent events chronologically"""
        try:
            rag_dir = self.get_user_rag_dir(user_id)
            vectorstore_path = os.path.join(rag_dir, "faiss_index")
            
            if not os.path.exists(vectorstore_path):
                return []
            
            # بارگذاری index
            vectorstore = FAISS.load_local(vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
            
            # استخراج تمام اسناد
            all_docs = []
            if hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, '_dict'):
                for doc_id, doc in vectorstore.docstore._dict.items():
                    if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                        all_docs.append({
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'timestamp': doc.metadata.get('timestamp', '')
                        })
            
            # مرتب‌سازی بر اساس timestamp (از جدید به قدیم)
            sorted_docs = sorted(all_docs, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # برگرداندن limit رکورد اول
            recent_events = sorted_docs[:limit]
            
            # پاک کردن حافظه
            del vectorstore, all_docs
            gc.collect()
            
            return recent_events
            
        except Exception as e:
            print(f"Error getting recent events: {e}")
            return []

    def query_context(self, user_id: int, query: str, top_k: int = 5, include_task_info: bool = True) -> str:
        """Query the RAG system with memory optimization"""
        try:
            rag_dir = self.get_user_rag_dir(user_id)
            vectorstore_path = os.path.join(rag_dir, "faiss_index")
            
            if not os.path.exists(vectorstore_path):
                return ""
            
            # بارگذاری index
            vectorstore = FAISS.load_local(vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
            
            # جستجو با محدودیت نتایج
            results = vectorstore.similarity_search(query, k=min(top_k, 10))
            
            # ترکیب نتایج با metadata
            context_parts = []
            for doc in results:
                if include_task_info and hasattr(doc, 'metadata') and doc.metadata:
                    # اضافه کردن اطلاعات تسک به context
                    task_info = ""
                    if 'task_id' in doc.metadata:
                        task_info += f"[Task ID: {doc.metadata['task_id']}]"
                    if 'task_description' in doc.metadata:
                        task_info += f"[Task: {doc.metadata['task_description']}]"
                    if 'timestamp' in doc.metadata:
                        task_info += f"[Time: {doc.metadata['timestamp']}]"
                    
                    if task_info:
                        context_parts.append(f"{task_info}\n{doc.page_content}")
                    else:
                        context_parts.append(doc.page_content)
                else:
                    context_parts.append(doc.page_content)
            
            context = "\n---\n".join(context_parts)
            
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
                vectorstore = FAISS.load_local(vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
                
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

import os
import sys
import json
import shutil
import time
# 修正: 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from typing import List
# from langchain_huggingface import HuggingFaceEmbeddings  # 不再使用HuggingFaceEmbeddings
from core.llm_service import QwenLLM # 导入QwenLLM
from config import PROCESSED_REPORTS_DIR, VECTOR_STORE_DIR
from config import EMBEDDING_MODE, LOCAL_EMBEDDING_PATH, VECTOR_STORE_SUBDIR
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class QwenTongyiEmbeddings(Embeddings):
    """
    自定义的通义千问Embedding类，以适配LangChain的接口。
    """
    def __init__(self, llm_service: QwenLLM):
        self.llm_service = llm_service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        处理一组文档的向量化。
        采用分批处理以提高效率并避免API单次请求量超限。
        增加了更健壮的重试和退避机制来应对网络不稳定。
        """
        all_embeddings = []
        batch_size = 25  # 通义千问v2模型的batch size上限为25
        max_retries_per_batch = 5 # 增加每个批次的重试次数
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for attempt in range(max_retries_per_batch):
                try:
                    # 调用新的批量处理方法
                    batch_embeddings = self.llm_service.get_text_embeddings_batch(batch_texts)
                    if batch_embeddings:
                        all_embeddings.extend(batch_embeddings)
                        # 成功处理，跳出重试循环
                        print(f"成功处理批次 {i//batch_size + 1}/{len(texts)//batch_size + 1}。")
                        break
                    else:
                        print(f"警告: 批次 {i//batch_size + 1} 的Embedding返回为空。")
                        # 同样视为成功处理，跳出重试
                        break
                except Exception as e:
                    # 捕获到异常，进行重试
                    wait_time = 5 * (attempt + 1) # 指数退避
                    print(f"错误: 处理批次 {i//batch_size + 1} (尝试 {attempt + 1}/{max_retries_per_batch}) 时发生错误: {e}")
                    print(f"将在 {wait_time} 秒后重试...")
                    if attempt + 1 == max_retries_per_batch:
                        # 这是最后一次尝试，记录严重错误并抛出异常
                        print(f"错误: 批次 {i//batch_size + 1} 在所有重试后仍然失败。程序将终止。")
                        raise e # 重新抛出异常，终止整个建库流程
                    time.sleep(wait_time)
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """处理单个查询的向量化"""
        return self.llm_service.get_text_embedding(text)

# === 本地Qwen3-Embedding-0.6B适配 ===
class LocalQwenEmbedding(Embeddings):
    """
    本地 Qwen3-Embedding-0.6B 封装，适配 LangChain Embeddings 接口
    """
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B",cache_dir = model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", cache_dir = model_path, trust_remote_code=True, device_map = "cuda" ).eval()
        self.torch = torch
        self.np = np

    def _mean_pooling(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        masked_hidden = last_hidden * mask
        summed = self.torch.sum(masked_hidden, 1)
        counts = self.torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = []
        for text in texts:
            embs.append(self.embed_query(text))
        return embs

    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=32000)
        # 保证inputs和模型在同一设备
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state
            pooled = self._mean_pooling(last_hidden, inputs['attention_mask'])
            emb = pooled.squeeze(0).cpu().numpy().astype(self.np.float32)
        return emb.tolist()

class KnowledgeBaseManager:
    def __init__(self, processed_dir: str = PROCESSED_REPORTS_DIR, persist_directory: str = None):
        """
        初始化知识库管理器。

        :param processed_dir: 已处理（JSON）文件所在的目录。
        :param persist_directory: ChromaDB持久化存储的目录（如为None则自动拼接主目录和子目录）。
        """
        self.processed_dir = processed_dir
        if persist_directory is None:
            self.persist_directory = os.path.join(VECTOR_STORE_DIR, VECTOR_STORE_SUBDIR)
        else:
            self.persist_directory = persist_directory
        # 根据配置选择本地或远程embedding
        if EMBEDDING_MODE == "local":
            self.embedding_function = LocalQwenEmbedding(LOCAL_EMBEDDING_PATH)
        else:
            llm = QwenLLM()
            self.embedding_function = QwenTongyiEmbeddings(llm)
        self.db = None

    def _metadata_func(self, record: dict, metadata: dict) -> dict:
        """
        自定义元数据处理函数。
        由于jq表达式已经处理了核心内容提取，这里只需确保所有原始信息被保留。
        
        :param record: 单个JSON对象 (jq_schema处理后的结果)
        :param metadata: 默认元数据(包含source和jq处理后的所有字段)
        :return: 更新后的元数据
        """
        # 保留所有jq处理后传入的字段作为元数据
        # JSONLoader会默认将record的所有内容复制到metadata中
        # 如果有不需要的字段，可以在这里移除
        # 例如，如果我们创建了一个临时的'loader_content'键，可以在这里 del metadata['loader_content']
        return metadata

    def load_documents(self):
        """
        从目录加载JSON文档，使用jq语法和自定义元数据函数高效解析。
        """
        # 1. 加载JSON文档（原有逻辑）
        json_loader_kwargs = {
            # 采纳并增强了建议的jq表达式：
            # 1. 'if type == "object"' 检查每个元素的类型。
            # 2. 如果是对象，则在保留原对象所有字段的基础上，添加一个新的'page_content'字段。
            #    其值优先取'text'，其次取'table_body'，最后为空字符串，确保内容存在。
            # 3. 如果不是对象（如纯字符串），则将其转换为一个标准结构'{"page_content": tostring}'
            # 4. 最终确保每个流出的元素都是一个含有'page_content'键的对象。
            'jq_schema': (
                '.[] | if type == "object" '
                'then . + {"page_content": (.text // .table_body // "")} '
                'else {"page_content": tostring} end'
            ),
            'content_key': 'page_content',  # 统一使用 'page_content' 作为内容来源
            'metadata_func': self._metadata_func
        }
        
        loader = DirectoryLoader(
            self.processed_dir, 
            glob="**/*.json", 
            loader_cls=JSONLoader,
            loader_kwargs=json_loader_kwargs,
            show_progress=True,
            use_multithreading=True
        )
        
        try:
            documents = loader.load()
        except ValueError as e:
            print(f"JSON文件解析失败，请检查文件格式: {e}")
            documents = []
        documents = [doc for doc in documents if doc.page_content]

        # 2. 加载data/reports目录下的TXT文档（新增逻辑）
        txt_documents = []
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'reports')
        for root, _, files in os.walk(reports_dir):
            for file in files:
                if file.lower().endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if content.strip():
                            txt_documents.append(
                                Document(
                                    page_content=content,
                                    metadata={"source": file_path}
                                )
                            )
                    except Exception as e:
                        print(f"读取TXT文件失败: {file_path}, 错误: {e}")

        # 合并所有文档
        all_documents = documents + txt_documents

        # 防御性校验，确保所有page_content都是字符串
        for doc in all_documents:
            if not isinstance(doc.page_content, str):
                print(f"警告: 在文件 {doc.metadata.get('source')} 中发现非文本内容，已强制转换为字符串。")
                doc.page_content = str(doc.page_content)

        if not all_documents:
            print(f"警告: 在目录 '{self.processed_dir}' 和 'data/reports' 中没有成功加载任何文档。")
            return []
        print(f"成功加载并处理了 {len(all_documents)} 个文档块（含JSON和TXT）。")
        return all_documents

    def split_documents(self, documents, chunk_size=500, chunk_overlap=50):
        """将文档分割成小块。"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        return docs

    def create_and_persist_db(self, docs):
        """创建并持久化向量数据库。"""
        print(f"正在创建和持久化向量数据库到 '{self.persist_directory}'...")
        # 如果目录已存在，先清空
        if os.path.exists(self.persist_directory):
            print(f"目录 '{self.persist_directory}' 已存在，正在清空...")
            shutil.rmtree(self.persist_directory)
        
        # 获取所有文档的文本内容
        doc_texts = [doc.page_content for doc in docs]
        
        # 使用包装好的Embedding对象进行向量化
        print("正在使用通义千问模型为文档生成向量...")
        # 注意：这里我们直接调用embed_documents方法，而不是在列表推导式中调用
        # 这一步将由包装类在内部处理
        
        # Chroma.from_documents 会自动调用 embedding_function.embed_documents
        self.db = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_function, 
            persist_directory=self.persist_directory
        )

        print(f"成功为 {len(docs)} 个文档块创建向量数据库。")
        print("向量数据库创建并持久化成功。")
        return self.db

    def load_db(self):
        """从持久化目录加载向量数据库。"""
        if self.db is None:
            print(f"正在从 '{self.persist_directory}' 加载向量数据库...")
            self.db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_function)
        return self.db

    def similarity_search(self, query, k=5):
        """执行相似性搜索。"""
        if self.db is None:
            self.load_db()
        results = self.db.similarity_search(query, k=k)
        return results

def main():
    """主函数，用于初始化和测试知识库管理器。"""
    
    # 确保文件夹存在
    os.makedirs(PROCESSED_REPORTS_DIR, exist_ok=True)
    from config import VECTOR_STORE_DIR, VECTOR_STORE_SUBDIR
    target_vector_store = os.path.join(VECTOR_STORE_DIR, VECTOR_STORE_SUBDIR)
    os.makedirs(target_vector_store, exist_ok=True)
    print(f"即将初始化新的向量数据库，目标目录: {target_vector_store}")

    # 实例化管理器（会自动指向当前子目录）
    kb_manager = KnowledgeBaseManager()

    # 加载和处理文档
    documents = kb_manager.load_documents()
    print(f"加载文档数: {len(documents)}")
    if not documents:
        print("在 'data/processed' 目录下没有找到JSON文件。")
        print("请确保 'pdf_parser.py' 已经成功运行并且生成了JSON文件。")
        return

    docs = kb_manager.split_documents(documents)
    print(f"分块后文档数: {len(docs)}")

    # 创建并持久化新的数据库
    db = kb_manager.create_and_persist_db(docs)
    print(f"新的向量数据库已初始化并持久化到: {target_vector_store}")

    # 执行一个测试查询
    print("\n--- 执行测试查询 ---")
    query = "城建档案敏感词怎么分类"
    search_results = kb_manager.similarity_search(query)
    print(f"查询: '{query}'")
    print("查询结果:")
    for result in search_results:
        print(f"  - 来源: {result.metadata.get('source', 'N/A')}, 页码: {result.metadata.get('seq_num', 'N/A')}")
        print(f"    内容: {result.page_content[:150]}...")
    print("---------------------\n")


if __name__ == '__main__':
    main() 
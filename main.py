# -*- coding: utf-8 -*-
"""
@author: hmd
@license: (C) Copyright 2021-2027, JMD.
@contact: 931725379@qq.com
@software: 
@file: main.py
@time: 2024/7/29 11:00
@desc: RAG档案文件开放审核应用的主入口，使用FastAPI提供Web服务
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import os
import shutil
from fastapi.staticfiles import StaticFiles
import datetime
import json
import glob

# 导入我们的核心服务
from core.qa_service import QAService
from core.archives_manager import ArchivesManager
from core.rulers_manager import RulersManager
# 创建档案管理器实例
archives_manager = ArchivesManager()
rulers_manager = RulersManager()

# --- 数据模型定义 ---
class AskRequest(BaseModel):
    query: str
    top_k: int = 20
    rerank_top_n: int = 5

class AuditRequest(BaseModel):
    file_names: list[str]
    top_k: int = 20
    rerank_top_n: int = 5


# --- 应用生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行
    print("应用启动... 正在初始化QA服务...")
    app.state.qa_service = QAService()
    print("QA服务初始化完成。")
    
    yield
    
    # 应用关闭时执行 (如果需要)
    print("应用关闭。")


# --- FastAPI应用初始化 ---
app = FastAPI(
    title="RAG 档案文件开放审核 API",
    description="一个基于RAG架构的档案文件开放审核API服务。",
    version="1.0.0",
    lifespan=lifespan  # 使用新的lifespan参数
)

# 挂载档案目录为静态文件服务
archives_static_dir = os.path.join(os.path.dirname(__file__), "data", "archives")
app.mount("/static/archives", StaticFiles(directory=archives_static_dir), name="archives-static")

# 挂载审核规则目录为静态文件服务
reports_static_dir = os.path.join(os.path.dirname(__file__), "data", "reports")
app.mount("/static/reports", StaticFiles(directory=reports_static_dir), name="reports-static")

# --- 中间件配置 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API路由定义 ---
@app.get("/", summary="根路径")
async def root():
    return {"message": "欢迎使用RAG档案文件开放审核API！服务运行正常。"}

# --- 问答API ---
@app.post("/api/ask", summary="执行完整的RAG问答流程")
async def ask_question(request: AskRequest):
    """
    接收档案文件，执行完整的检索、重排和生成流程，返回结构化的答案。
    """
    qa_service: QAService = app.state.qa_service
    # 注意：我们将在这里直接调用一个非流式的ask方法
    result = qa_service.ask(
        query=request.query, 
        top_k=request.top_k, 
        rerank_top_n=request.rerank_top_n
    )
    return result

# --- 档案文件审核API ---
@app.post("/api/audit", summary="对档案文件进行RAG批量审核")
async def audit_files(request: AuditRequest):
    qa_service: QAService = app.state.qa_service
    contents = []
    file_names = []
    for file_name in request.file_names:
        try:
            content = archives_manager.view_file(file_name)
            if not content:
                contents.append(None)
            else:
                contents.append(content)
            file_names.append(file_name)
        except Exception as e:
            contents.append(None)
            file_names.append(file_name)
    # 一次性传入所有内容
    batch_results = qa_service.batch_audit(contents, top_k=request.top_k, rerank_top_n=request.rerank_top_n)
    results = []
    for fn, c, seg_results in zip(file_names, contents, batch_results):
        if not c:
            results.append({
                "file_name": fn,
                "raw_content": "",
                "result": [{"error": "内容为空或读取失败"}]
            })
        else:
            # 保证seg_results为数组
            if not isinstance(seg_results, list):
                seg_results = [seg_results]
            results.append({
                "file_name": fn,
                "raw_content": c[:200],
                "result": seg_results
            })
    # 保存审核结果
    save_dir = os.path.join(os.path.dirname(__file__), "data", "audit_results")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"audit_{timestamp}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return {"results": results, "saved_to": save_path}

@app.get("/api/audit/last_result", summary="获取最近一次批量审核结果")
async def get_last_audit_result():
    result_dir = os.path.join(os.path.dirname(__file__), "data", "audit_results")
    files = sorted(glob.glob(os.path.join(result_dir, "audit_*.json")), reverse=True)
    if not files:
        return {"results": []}
    with open(files[0], "r", encoding="utf-8") as f:
        results = json.load(f)
    return {"results": results}

@app.get("/api/audit/all_results", summary="获取所有历史批量审核结果")
async def get_all_audit_results():
    import glob
    result_dir = os.path.join(os.path.dirname(__file__), "data", "audit_results")
    files = sorted(glob.glob(os.path.join(result_dir, "audit_*.json")), reverse=True)
    all_results = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                all_results.append({"file": os.path.basename(fpath), "results": results})
            except Exception:
                continue
    return {"all_results": all_results}

# --- 档案管理API ---
@app.post("/api/archives/upload", summary="批量上传档案到档案库")
async def upload_archives(files: list[UploadFile] = File(...)):
    uploaded = []
    temp_files = []
    try:
        for file in files:
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_files.append(temp_path)
        uploaded = archives_manager.upload_files(temp_files)
        return {"uploaded": uploaded}
    finally:
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.get("/api/archives/list", summary="获取archives目录下所有档案名")
async def list_archives():
    return {"files": archives_manager.list_files()}

@app.get("/api/archives/view", summary="查看指定档案内容")
async def view_archive(file_name: str = ""):
    if not file_name:
        raise HTTPException(status_code=400, detail="file_name参数不能为空")
    content = archives_manager.view_file(file_name)
    if content is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"content": content}

@app.delete("/api/archives/delete", summary="删除指定档案")
async def delete_archive(file_name: str = ""):
    if not file_name:
        raise HTTPException(status_code=400, detail="file_name参数不能为空")
    result = archives_manager.delete_file(file_name)
    if not result:
        raise HTTPException(status_code=404, detail="文件不存在或删除失败")
    return {"deleted": file_name}

# --- 审核规则管理API ---
@app.post("/api/reports/upload", summary="批量上传审核规则文件到reports目录")
async def upload_reports(files: list[UploadFile] = File(...)):
    uploaded = []
    temp_files = []
    try:
        for file in files:
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_files.append(temp_path)
        uploaded = rulers_manager.upload_files(temp_files)
        return {"uploaded": uploaded}
    finally:
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.get("/api/reports/list", summary="获取reports目录下所有审核规则文件名")
async def list_reports():
    return {"files": rulers_manager.list_files()}

    
@app.get("/api/reports/view", summary="查看指定审核规则文件内容")
async def view_report(file_name: str = ""):
    if not file_name:
        raise HTTPException(status_code=400, detail="file_name参数不能为空")
    content = rulers_manager.view_file(file_name)
    if content is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"content": content}

@app.delete("/api/reports/delete", summary="删除指定审核规则文件")
async def delete_report(file_name: str = ""):
    if not file_name:
        raise HTTPException(status_code=400, detail="file_name参数不能为空")
    result = rulers_manager.delete_file(file_name)
    if not result:
        raise HTTPException(status_code=404, detail="文件不存在或删除失败")
    return {"deleted": file_name}

# --- 启动服务 ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
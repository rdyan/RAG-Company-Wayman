import os
import shutil
import json
from typing import List, Optional
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from paddleocr import PaddleOCR
    import numpy as np
    from PIL import Image
except ImportError:
    PaddleOCR = None
    np = None
    Image = None

ARCHIVES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'archives')

class ArchivesManager:
    def __init__(self, archives_dir: Optional[str] = None):
        self.archives_dir = archives_dir or ARCHIVES_DIR
        os.makedirs(self.archives_dir, exist_ok=True)
        self._ocr_engine = None

    def _get_ocr_engine(self):
        if self._ocr_engine is None:
            if PaddleOCR:
                print(" -> Initializing PaddleOCR engine (this may take a moment on first run)...")
                self._ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
                print(" -> PaddleOCR engine initialized.")
            else:
                return None
        return self._ocr_engine

    def upload_files(self, file_paths: List[str]) -> List[str]:
        uploaded = []
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                continue
            dest_path = os.path.join(self.archives_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
            uploaded.append(os.path.basename(file_path))
        return uploaded

    def view_file(self, file_name: str) -> Optional[str]:
        file_path = os.path.join(self.archives_dir, file_name)
        if not os.path.isfile(file_path):
            return None
        ext = Path(file_path).suffix.lower()
        try:
            if ext == '.pdf':
                if fitz is None:
                    return "PyMuPDF (fitz) is not installed."
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                if len(text.strip()) < 100:
                    print(f" -> Standard text extraction for '{file_name}' was insufficient. Attempting OCR with PaddleOCR...")
                    ocr_engine = self._get_ocr_engine()
                    if ocr_engine is None or np is None or Image is None:
                        return "OCR libraries (paddleocr, numpy, Pillow) not installed, but may be required for this PDF."
                    ocr_texts = []
                    for i, page in enumerate(doc):
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img_array = np.array(img)
                        result = ocr_engine.ocr(img_array, cls=True)
                        if result and result[0] is not None:
                            text_from_page = '\n'.join([line[1][0] for line in result[0]])
                            ocr_texts.append(text_from_page)
                    ocr_full_text = '\n'.join(ocr_texts).strip()
                    if len(ocr_full_text) > len(text):
                        text = ocr_full_text
                return text
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2)
            else:
                return f"不支持的文件格式: {ext}"
        except Exception as e:
            return f"读取文件时出错: {e}"

    def delete_file(self, file_name: str) -> bool:
        file_path = os.path.join(self.archives_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            return True
        return False

    def list_files(self) -> List[str]:
        return [f for f in os.listdir(self.archives_dir) if os.path.isfile(os.path.join(self.archives_dir, f))]

if __name__ == '__main__':
    # 示例：创建实例
    manager = ArchivesManager()
    print('当前 archives 目录下的文件:')
    print(manager.list_files())

    # 示例：批量上传（请替换为你本地实际存在的文件路径）
    # uploaded = manager.upload_files(['test1.pdf', 'test2.txt'])
    # print('上传后的文件:')
    # print(manager.list_files())

    # 示例：查看文件内容（请替换为实际文件名）
    # content = manager.view_file('test1.pdf')
    # print('文件内容:')
    # print(content)

    # 示例：删除文件（请替换为实际文件名）
    # result = manager.delete_file('test1.pdf')
    # print('删除结果:', result)
    # print('删除后的文件:')
    # print(manager.list_files())

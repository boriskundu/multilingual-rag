"""
Process government health data - FDA + MedlinePlus (NIH) ONLY
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

class HealthDataProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_json_file(self, file_path: Path) -> List[Dict]:
        """Load JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if 'results' in data:  # FDA format
                return data['results']
            else:
                return [data]  # MedlinePlus format
        elif isinstance(data, list):
            return data
        
        return [data]
    
    def process_fda_data(self, file_path: Path) -> List[Document]:
        """Process FDA drug data (U.S. Government)"""
        documents = []
        data = self.load_json_file(file_path)
        
        for item in data:
            if isinstance(item, dict):
                metadata = {
                    "source": "FDA",
                    "source_type": "U.S. Government",
                    "file": file_path.name,
                    "drug_name": item.get("openfda", {}).get("brand_name", ["Unknown"])[0]
                    if "openfda" in item else "Unknown"
                }
                
                text_parts = []
                fields = [
                    "indications_and_usage",
                    "dosage_and_administration", 
                    "warnings",
                    "warnings_and_cautions",
                    "adverse_reactions",
                    "description",
                    "clinical_pharmacology"
                ]
                
                for field in fields:
                    if field in item and item[field]:
                        field_content = item[field][0] if isinstance(item[field], list) else item[field]
                        text_parts.append(f"{field.upper().replace('_', ' ')}: {field_content}")
                
                if text_parts:
                    content = "\n\n".join(text_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
        
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        
        logger.info(f"Processed {len(data)} FDA records into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def process_medlineplus_data(self, file_path: Path) -> List[Document]:
        """Process MedlinePlus data (NIH - U.S. Government)"""
        documents = []
        data = self.load_json_file(file_path)
        
        for item in data:
            if isinstance(item, dict):
                metadata = {
                    "source": "MedlinePlus (NIH)",
                    "source_type": "U.S. Government",
                    "file": file_path.name,
                    "topic": item.get("topic", "Unknown"),
                    "url": item.get("url", "")
                }
                
                text_parts = []
                
                if item.get('summary'):
                    text_parts.append(f"SUMMARY: {item['summary']}")
                
                if item.get('sections'):
                    for section_title, section_content in item['sections'].items():
                        text_parts.append(f"{section_title}: {section_content}")
                
                if not text_parts and item.get('full_text'):
                    text_parts.append(item['full_text'])
                
                if text_parts:
                    content = "\n\n".join(text_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
        
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        
        logger.info(f"Processed {len(data)} MedlinePlus topics into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def process_all_files(self, data_dir: Path) -> List[Document]:
        """Process all government data files"""
        all_documents = []
        
        # Process FDA files
        fda_dir = data_dir / "fda"
        if fda_dir.exists():
            for file_path in fda_dir.glob("*.json"):
                logger.info(f"Processing FDA (U.S. Gov): {file_path.name}")
                docs = self.process_fda_data(file_path)
                all_documents.extend(docs)
        
        # Process MedlinePlus files
        medlineplus_dir = data_dir / "medlineplus"
        if medlineplus_dir.exists():
            for file_path in medlineplus_dir.glob("*.json"):
                logger.info(f"Processing MedlinePlus (NIH): {file_path.name}")
                docs = self.process_medlineplus_data(file_path)
                all_documents.extend(docs)
        
        logger.info(f"Total government documents processed: {len(all_documents)}")
        return all_documents
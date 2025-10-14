"""
Process and chunk health data for RAG system
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
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
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if 'results' in data:  # FDA format
                return data['results']
            elif 'data' in data:  # CDC download format
                return data['data']
            elif 'esearchresult' in data:  # PubMed format
                return [data['esearchresult']]
        elif isinstance(data, list):
            return data
        
        return [data]
    
    def process_cdc_data(self, file_path: Path) -> List[Document]:
        """Process CDC JSON data into documents"""
        documents = []
        data = self.load_json_file(file_path)
        
        for item in data:
            # Extract text content from CDC data
            if isinstance(item, dict):
                text_parts = []
                metadata = {"source": "CDC", "file": file_path.name}
                
                for key, value in item.items():
                    if value and str(value).strip():
                        text_parts.append(f"{key}: {value}")
                        if key in ["year", "state", "category"]:
                            metadata[key] = str(value)
                
                if text_parts:
                    content = "\n".join(text_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
        
        # Split documents into chunks
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        
        logger.info(f"Processed {len(data)} CDC records into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def process_fda_data(self, file_path: Path) -> List[Document]:
        """Process FDA drug data into documents"""
        documents = []
        data = self.load_json_file(file_path)
        
        for item in data:
            if isinstance(item, dict):
                metadata = {
                    "source": "FDA",
                    "file": file_path.name,
                    "drug_name": item.get("openfda", {}).get("brand_name", ["Unknown"])[0]
                    if "openfda" in item else "Unknown"
                }
                
                # Extract relevant fields
                text_parts = []
                fields = ["indications_and_usage", "dosage_and_administration", 
                         "warnings", "description", "clinical_pharmacology"]
                
                for field in fields:
                    if field in item and item[field]:
                        text_parts.append(f"{field.upper()}: {item[field][0] if isinstance(item[field], list) else item[field]}")
                
                if text_parts:
                    content = "\n\n".join(text_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
        
        # Split documents into chunks
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        
        logger.info(f"Processed {len(data)} FDA records into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def process_all_files(self, data_dir: Path) -> List[Document]:
        """Process all files in data directory"""
        all_documents = []
        
        # Process CDC files
        cdc_dir = data_dir / "cdc"
        if cdc_dir.exists():
            for file_path in cdc_dir.glob("*.json"):
                logger.info(f"Processing CDC file: {file_path.name}")
                docs = self.process_cdc_data(file_path)
                all_documents.extend(docs)
        
        # Process FDA files
        fda_dir = data_dir / "fda"
        if fda_dir.exists():
            for file_path in fda_dir.glob("*.json"):
                logger.info(f"Processing FDA file: {file_path.name}")
                docs = self.process_fda_data(file_path)
                all_documents.extend(docs)
        
        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents
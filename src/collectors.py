"""
Data collectors for healthcare RAG research
"""
import requests
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class HealthDataCollector:
    """Base class for health data collection"""
    
    def __init__(self, base_url: str, data_dir: Path):
        self.base_url = base_url
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch(self, endpoint: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Fetch data from API endpoint"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {str(e)}")
            return None
    
    def save(self, data: Dict, filename: str):
        """Save data to JSON file"""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved data to {filepath}")
        return filepath

class CDCCollector(HealthDataCollector):
    """CDC specific data collector"""
    
    def __init__(self, data_dir: Path):
        super().__init__("https://data.cdc.gov/", data_dir / "cdc")
        self.app_token = os.getenv("CDC_APP_TOKEN")
        self.headers = {"X-App-Token": self.app_token} if self.app_token else {}
        
    def fetch_diabetes_data(self, limit: int = 1000):
        """Fetch diabetes statistics from CDC"""
        data = self.fetch("resource/fn2i-3j6c.json", {"$limit": limit}, self.headers)
        if data:
            return self.save(data, "diabetes_stats.json")
        return None
    
    def fetch_vaccination_data(self, limit: int = 1000):
        """Fetch vaccination data"""
        # Try download format for better compatibility
        data = self.fetch("api/views/unsk-b7fc/rows.json", 
                         {"accessType": "DOWNLOAD"}, self.headers)
        if data:
            return self.save(data, "vaccinations.json")
        return None
    
    def fetch_heart_disease_data(self, limit: int = 1000):
        """Fetch heart disease mortality data"""
        data = self.fetch("resource/jiwm-ppbh.json", {"$limit": limit}, self.headers)
        if data:
            return self.save(data, "heart_disease.json")
        return None

class FDACollector(HealthDataCollector):
    """FDA specific data collector"""
    
    def __init__(self, data_dir: Path):
        super().__init__("https://api.fda.gov/", data_dir / "fda")
    
    def fetch_drug_labels(self, search_term: str, limit: int = 100):
        """Fetch drug label information"""
        data = self.fetch("drug/label.json", {
            "search": search_term,
            "limit": limit
        })
        if data:
            filename = f"{search_term.replace(' ', '_')}_drugs.json"
            return self.save(data, filename)
        return None

class PubMedCollector(HealthDataCollector):
    """PubMed literature collector"""
    
    def __init__(self, data_dir: Path):
        super().__init__("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/", 
                        data_dir / "pubmed")
    
    def search_articles(self, query: str, max_results: int = 50):
        """Search PubMed for articles"""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        data = self.fetch("esearch.fcgi", params)
        if data:
            filename = f"{query.replace(' ', '_')}_search.json"
            return self.save(data, filename)
        return None
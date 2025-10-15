"""
Data collectors - GOVERNMENT SOURCES ONLY
FDA (fda.gov) + MedlinePlus/NIH (nlm.nih.gov)
"""
import requests
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup

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

class FDACollector(HealthDataCollector):
    """
    FDA Drug Labels - U.S. Food & Drug Administration
    Source: https://www.fda.gov
    API: https://open.fda.gov
    """
    
    def __init__(self, data_dir: Path):
        super().__init__("https://api.fda.gov/", data_dir / "fda")
    
    def fetch_drug_labels(self, search_term: str, limit: int = 100):
        """Fetch drug label information from FDA"""
        data = self.fetch("drug/label.json", {
            "search": search_term,
            "limit": limit
        })
        if data:
            filename = f"{search_term.replace(' ', '_')}_drugs.json"
            return self.save(data, filename)
        return None

class MedlinePlusCollector(HealthDataCollector):
    """
    MedlinePlus - National Library of Medicine (NIH)
    Source: https://medlineplus.gov (U.S. Government)
    Run by: National Institutes of Health (NIH)
    """
    
    def __init__(self, data_dir: Path):
        super().__init__("https://medlineplus.gov/", data_dir / "medlineplus")
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Research/Educational Use - Government Data)'
        }
    
    def scrape_health_topic(self, url: str, topic_name: str):
        """
        Scrape a MedlinePlus health topic page
        MedlinePlus is U.S. Government (public domain)
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            content = {
                'source': 'MedlinePlus (NIH)',
                'source_type': 'U.S. Government',
                'topic': topic_name,
                'url': url,
                'scraped_date': datetime.now().isoformat(),
                'summary': '',
                'sections': {}
            }
            
            # Extract summary
            summary_div = soup.find('div', {'id': 'topic-summary'})
            if summary_div:
                content['summary'] = summary_div.get_text(strip=True)
            
            # Extract main content sections
            main_content = soup.find('div', {'id': 'mplus-content'})
            if main_content:
                sections = main_content.find_all('section')
                for section in sections:
                    heading = section.find(['h2', 'h3'])
                    if heading:
                        section_title = heading.get_text(strip=True)
                        paragraphs = section.find_all('p')
                        section_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                        if section_text:
                            content['sections'][section_title] = section_text
            
            # Fallback: get all paragraphs if no sections found
            if not content['sections']:
                all_paragraphs = soup.find_all('p')
                content['full_text'] = ' '.join([p.get_text(strip=True) for p in all_paragraphs])
            
            filename = f"{topic_name.replace(' ', '_').lower()}.json"
            filepath = self.save(content, filename)
            
            logger.info(f"✓ Scraped government source: {topic_name}")
            time.sleep(2)  # Be respectful to government servers
            return filepath
            
        except Exception as e:
            logger.error(f"Error scraping MedlinePlus {topic_name}: {e}")
            return None
    
    def fetch_multiple_topics(self, topics: List[tuple]):
        """
        Fetch multiple health topics from MedlinePlus
        topics: List of (url, topic_name) tuples
        """
        results = []
        logger.info(f"Collecting from MedlinePlus (NIH - U.S. Government)")
        for url, topic_name in topics:
            logger.info(f"  Fetching: {topic_name}")
            filepath = self.scrape_health_topic(url, topic_name)
            if filepath:
                results.append(filepath)
        return results
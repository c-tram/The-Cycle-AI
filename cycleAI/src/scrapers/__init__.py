"""
Web scraping components for data extraction.
"""

from .js_scraper import SyncJavaScriptScraper
from .query_driven_scraper import SyncQueryDrivenScraper

__all__ = [
    'SyncJavaScriptScraper',
    'SyncQueryDrivenScraper'
]

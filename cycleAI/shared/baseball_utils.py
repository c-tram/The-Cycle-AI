"""
Shared Baseball Data Manager and Utilities
Used across Simple, Advanced, and Autonomous modes
"""

import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib

class BaseballDataManager:
    """Shared data management with caching for all modes"""

    def __init__(self, cache_duration_minutes: int = 30):
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.cache = {}
        self.team_league_map = {
            'BAL': 'AL', 'BOS': 'AL', 'NYY': 'AL', 'TB': 'AL', 'TOR': 'AL',
            'CLE': 'AL', 'DET': 'AL', 'MIN': 'AL', 'CWS': 'AL', 'KC': 'AL',
            'LAA': 'AL', 'OAK': 'AL', 'SEA': 'AL', 'TEX': 'AL', 'HOU': 'AL',
            'ATL': 'NL', 'MIA': 'NL', 'NYM': 'NL', 'PHI': 'NL', 'WSH': 'NL',
            'CHC': 'NL', 'CIN': 'NL', 'MIL': 'NL', 'PIT': 'NL', 'STL': 'NL',
            'ARI': 'NL', 'COL': 'NL', 'LAD': 'NL', 'SD': 'NL', 'SF': 'NL'
        }

    def _get_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        return datetime.now() - cache_entry['timestamp'] < self.cache_duration

    def fetch_data(self, url: str) -> Optional[Dict]:
        cache_key = self._get_cache_key(url)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            return self.cache[cache_key]['data']

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': datetime.now()
                }
                return data
        except Exception as e:
            print(f"❌ API error: {e}")
        return None

    def get_players_data(self) -> Optional[List[Dict]]:
        data = self.fetch_data("https://thecycle.online/api/v2/players")
        return data.get('players') if data else None

    def get_teams_data(self) -> Optional[List[Dict]]:
        data = self.fetch_data("https://thecycle.online/api/v2/teams")
        return data.get('teams') if data else None

    def get_team_league(self, team_code: str) -> str:
        return self.team_league_map.get(team_code, 'Unknown')

def setup_streamlit_config():
    """Shared Streamlit configuration"""
    import streamlit as st
    
    st.set_page_config(
        page_title="⚾ Baseball AI Agent",
        page_icon="⚾",
        layout="wide"
    )

def get_project_root():
    """Get the project root directory"""
    import os
    # Go up from shared/ to project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

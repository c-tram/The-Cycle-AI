import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@dataclass
class BaseballQuery:
    """Enhanced query analysis for better intent detection"""
    original_query: str
    is_player_query: bool
    is_team_query: bool
    requested_team: Optional[str]
    requested_stat: Optional[str]
    sort_direction: str = "desc"  # "asc" or "desc"
    min_games: Optional[int] = None
    league_filter: Optional[str] = None

class BaseballDataManager:
    """Enhanced data management with caching and preprocessing"""

    def __init__(self, cache_duration_minutes: int = 30):
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.cache = {}
        self.team_league_map = {
            'BAL': 'AL', 'BOS': 'AL', 'NYY': 'AL', 'TB': 'AL', 'TOR': 'AL',  # AL East
            'CLE': 'AL', 'DET': 'AL', 'MIN': 'AL', 'CWS': 'AL', 'KC': 'AL',   # AL Central
            'LAA': 'AL', 'OAK': 'AL', 'SEA': 'AL', 'TEX': 'AL', 'HOU': 'AL', # AL West
            'ATL': 'NL', 'MIA': 'NL', 'NYM': 'NL', 'PHI': 'NL', 'WSH': 'NL', # NL East
            'CHC': 'NL', 'CIN': 'NL', 'MIL': 'NL', 'PIT': 'NL', 'STL': 'NL', # NL Central
            'ARI': 'NL', 'COL': 'NL', 'LAD': 'NL', 'SD': 'NL', 'SF': 'NL'    # NL West
        }

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for API responses"""
        return hashlib.md5(url.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cached data is still valid"""
        return datetime.now() - cache_entry['timestamp'] < self.cache_duration

    def fetch_data(self, url: str) -> Optional[Dict]:
        """Fetch data with caching"""
        cache_key = self._get_cache_key(url)

        # Check cache first
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            print(f"📋 Using cached data for {url}")
            return self.cache[cache_key]['data']

        try:
            print(f"🌐 Fetching fresh data from {url}")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Cache the response
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': datetime.now()
                }
                return data
            else:
                print(f"❌ API failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ API error: {e}")
            return None

    def get_players_data(self) -> Optional[List[Dict]]:
        """Get players data with caching"""
        data = self.fetch_data("https://thecycle.online/api/v2/players")
        return data.get('players') if data else None

    def get_teams_data(self) -> Optional[List[Dict]]:
        """Get teams data with caching"""
        data = self.fetch_data("https://thecycle.online/api/v2/teams")
        return data.get('teams') if data else None

    def get_team_league(self, team_code: str) -> str:
        """Get league for a team code"""
        return self.team_league_map.get(team_code, 'Unknown')

class BaseballQueryAnalyzer:
    """Advanced query analysis for better intent detection"""

    def __init__(self):
        self.team_codes = [
            'HOU', 'NYY', 'BOS', 'BAL', 'TOR', 'TB', 'CLE', 'DET', 'MIN', 'CWS', 'KC',
            'LAA', 'OAK', 'SEA', 'TEX', 'ATL', 'MIA', 'NYM', 'PHI', 'WSH', 'MIL',
            'CHC', 'STL', 'PIT', 'CIN', 'ARI', 'COL', 'LAD', 'SD', 'SF'
        ]

        self.stat_mappings = {
            'hits': 'hits',
            'home runs': 'homeRuns', 'hr': 'homeRuns', 'homeruns': 'homeRuns',
            'doubles': 'doubles', '2b': 'doubles',
            'triples': 'triples', '3b': 'triples',
            'rbi': 'rbi', 'rbis': 'rbi',
            'runs': 'runs', 'r': 'runs',
            'avg': 'avg', 'average': 'avg', 'batting average': 'avg',
            'ops': 'ops',
            'slg': 'slg', 'slugging': 'slg',
            'obp': 'obp', 'on base': 'obp', 'on-base percentage': 'obp',
            'era': 'era', 'earned run average': 'era',
            'whip': 'whip',
            'strikeouts': 'strikeouts', 'k': 'strikeouts', 'so': 'strikeouts',
            'wins': 'wins', 'w': 'wins'
        }

    def analyze_query(self, query: str) -> BaseballQuery:
        """Analyze the user's query and extract intent"""
        query_lower = query.lower()

        # Determine query type
        is_player_query = any(keyword in query_lower for keyword in [
            'player', 'hr', 'home run', 'batting', 'rbi', 'avg', 'era',
            'strikeout', 'pitcher', 'hitter', 'catcher', 'outfielder', 'infielder'
        ])

        is_team_query = any(keyword in query_lower for keyword in [
            'team', 'teams', 'standing', 'record', 'win', 'loss', 'division',
            'league', 'rank', 'top', 'best', 'worst', 'al', 'nl', 'american', 'national'
        ])

        # If no specific type detected, fetch both
        if not is_player_query and not is_team_query:
            is_player_query = True
            is_team_query = True

        # Extract team
        requested_team = None
        for team_code in self.team_codes:
            if team_code.lower() in query_lower or f"{team_code.lower()} players" in query_lower:
                requested_team = team_code
                break

        # Extract stat
        requested_stat = None
        for stat_name, stat_key in self.stat_mappings.items():
            if (stat_name in query_lower or
                f"most {stat_name}" in query_lower or
                f"top {stat_name}" in query_lower or
                f"highest {stat_name}" in query_lower):
                requested_stat = stat_key
                break

        # Determine sort direction
        sort_direction = "desc"  # Default to descending (most to least)
        if any(word in query_lower for word in ['least', 'lowest', 'bottom', 'worst']):
            sort_direction = "asc"

        # Extract league filter
        league_filter = None
        if 'al' in query_lower or 'american league' in query_lower:
            league_filter = 'AL'
        elif 'nl' in query_lower or 'national league' in query_lower:
            league_filter = 'NL'

        # Extract minimum games (if mentioned)
        min_games = None
        import re
        games_match = re.search(r'(\d+)\s*games', query_lower)
        if games_match:
            min_games = int(games_match.group(1))

        return BaseballQuery(
            original_query=query,
            is_player_query=is_player_query,
            is_team_query=is_team_query,
            requested_team=requested_team,
            requested_stat=requested_stat,
            sort_direction=sort_direction,
            min_games=min_games,
            league_filter=league_filter
        )

class BaseballDataProcessor:
    """Process and filter baseball data based on query analysis"""

    def __init__(self, data_manager: BaseballDataManager):
        self.data_manager = data_manager

    def filter_players(self, players: List[Dict], query: BaseballQuery) -> List[Dict]:
        """Filter players based on query parameters"""
        filtered_players = []

        for player in players:
            if not isinstance(player, dict) or 'stats' not in player:
                continue

            # Team filter
            if query.requested_team and player.get('team') != query.requested_team:
                continue

            # League filter
            if query.league_filter:
                player_league = self.data_manager.get_team_league(player.get('team', ''))
                if player_league != query.league_filter:
                    continue

            # Minimum games filter
            if query.min_games:
                games = player.get('stats', {}).get('batting', {}).get('games', 0)
                if games < query.min_games:
                    continue

            # Stat availability filter (if specific stat requested)
            if query.requested_stat:
                if 'batting' in player['stats']:
                    stat_value = player['stats']['batting'].get(query.requested_stat, 0)
                    if stat_value == 0:
                        continue
                elif 'pitching' in player['stats']:
                    stat_value = player['stats']['pitching'].get(query.requested_stat, 0)
                    if stat_value == 0:
                        continue
                else:
                    continue

            filtered_players.append(player)

        return filtered_players

    def sort_players(self, players: List[Dict], query: BaseballQuery) -> List[Dict]:
        """Sort players based on query parameters"""
        if not query.requested_stat:
            return players

        def get_sort_key(player):
            # Try batting stats first, then pitching
            if 'batting' in player.get('stats', {}):
                return player['stats']['batting'].get(query.requested_stat, 0)
            elif 'pitching' in player.get('stats', {}):
                return player['stats']['pitching'].get(query.requested_stat, 0)
            return 0

        reverse = query.sort_direction == "desc"
        return sorted(players, key=get_sort_key, reverse=reverse)

    def format_player_data(self, player: Dict, query: BaseballQuery) -> str:
        """Format a single player's data for display"""
        name = player.get('name', 'Unknown')
        team = player.get('team', 'Unknown')
        position = player.get('position', 'Unknown')

        if query.requested_stat and 'stats' in player:
            if 'batting' in player['stats']:
                stat_value = player['stats']['batting'].get(query.requested_stat, 0)
                return f"{name} ({team}, {position}) | {stat_value} {query.requested_stat}"
            elif 'pitching' in player['stats']:
                stat_value = player['stats']['pitching'].get(query.requested_stat, 0)
                return f"{name} ({team}, {position}) | {stat_value} {query.requested_stat}"

        return f"{name} ({team}, {position})"

def answer_baseball_question_enhanced(query: str) -> str:
    """
    Enhanced version with advanced query analysis and data processing
    """
    print(f"🔍 Processing enhanced query: {query}")

    # Initialize components
    data_manager = BaseballDataManager()
    query_analyzer = BaseballQueryAnalyzer()
    data_processor = BaseballDataProcessor(data_manager)

    # Analyze the query
    parsed_query = query_analyzer.analyze_query(query)
    print(f"📋 Query analysis: team={parsed_query.requested_team}, stat={parsed_query.requested_stat}")

    # Fetch data based on query type
    all_data = {}

    if parsed_query.is_player_query:
        players = data_manager.get_players_data()
        if players:
            filtered_players = data_processor.filter_players(players, parsed_query)
            sorted_players = data_processor.sort_players(filtered_players, parsed_query)
            all_data['players'] = sorted_players[:20]  # Limit to top 20 for context
            print(f"✅ Processed {len(sorted_players)} players")

    if parsed_query.is_team_query:
        teams = data_manager.get_teams_data()
        if teams:
            all_data['teams'] = teams
            print(f"✅ Got {len(teams)} teams")

    # If we have no data, return error
    if not all_data:
        return "Sorry, I couldn't fetch the baseball data right now. Please try again later."

    # Build context for AI
    context_parts = []

    if 'players' in all_data:
        context_parts.append("PLAYERS DATA:")
        for player in all_data['players'][:10]:  # Show top 10
            context_parts.append(data_processor.format_player_data(player, parsed_query))

    if 'teams' in all_data:
        context_parts.append("\nTEAMS DATA:")
        for team in all_data['teams'][:5]:  # Show top 5 teams
            team_name = team.get('name', 'Unknown')
            wins = team.get('wins', 0)
            losses = team.get('losses', 0)
            context_parts.append(f"{team_name}: {wins}-{losses}")

    context = "\n".join(context_parts)

    # Use OpenAI to generate the final answer
    try:
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a baseball expert. Answer questions using the provided data. Be concise and accurate."},
                {"role": "user", "content": f"Question: {query}\n\nData:\n{context}"}
            ],
            max_tokens=500,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return f"Based on the data:\n\n{context}"

# Streamlit UI
def main():
    st.set_page_config(page_title="⚾ Baseball AI Agent", page_icon="⚾", layout="wide")

    st.title("⚾ Baseball AI Agent")
    st.markdown("*Smart baseball analytics using thecycle.online APIs*")

    # Sidebar with capabilities
    with st.sidebar:
        st.header("🎯 Capabilities")
        st.markdown("""
        **Team-Specific Queries:**
        - "HOU players by hits"
        - "AL teams by wins"
        - "NYY batting stats"

        **Stat Rankings:**
        - "most home runs"
        - "highest batting average"
        - "top RBI leaders"

        **Advanced Features:**
        - ✅ Smart query detection
        - ✅ Team & league filtering
        - ✅ Stat-based sorting
        - ✅ Real-time data
        - ✅ Comprehensive statistics
        """)

        st.header("📊 Data Sources")
        st.markdown("""
        - **Players API**: 1600+ MLB players
        - **Teams API**: 30 MLB teams
        - **Stats**: H, HR, RBI, AVG, OPS, ERA, WHIP, WAR, CVR
        - **Leagues**: AL vs NL identification
        """)

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about baseball..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing baseball data..."):
                response = answer_baseball_question_enhanced(prompt)
            st.markdown(response)

        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

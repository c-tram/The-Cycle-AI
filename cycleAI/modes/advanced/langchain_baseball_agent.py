"""
LangChain Integration for Baseball AI Agent
This shows how to integrate LangChain into your existing baseball agent
"""

import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import re

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.baseball_utils import BaseballDataManager, setup_streamlit_config

# LangChain imports
from langchain.tools import BaseTool, Tool
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@dataclass
class BaseballData:
    """Container for baseball data"""
    players: List[Dict] = None
    teams: List[Dict] = None

    def __post_init__(self):
        if self.players is None:
            self.players = []
        if self.teams is None:
            self.teams = []

class BaseballDataManager:
    """Enhanced data management with caching"""

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

# LangChain Tools
class FetchPlayersTool(BaseTool):
    """LangChain tool for fetching player data"""

    name: str = "fetch_players_data"
    description: str = "Fetch comprehensive player statistics from the MLB players API. Use this to get all player data for analysis."
    data_manager: BaseballDataManager

    def __init__(self, data_manager: BaseballDataManager):
        super().__init__(data_manager=data_manager)

    def _run(self, query: str = "") -> str:
        """Fetch players data"""
        players = self.data_manager.get_players_data()
        if players:
            return f"Successfully fetched {len(players)} players with complete statistics including batting, pitching, and biographical data."
        return "Failed to fetch players data. Please try again."

class FetchTeamsTool(BaseTool):
    """LangChain tool for fetching team data"""

    name: str = "fetch_teams_data"
    description: str = "Fetch team records and statistics from the MLB teams API. Use this to get team standings and performance data."
    data_manager: BaseballDataManager

    def __init__(self, data_manager: BaseballDataManager):
        super().__init__(data_manager=data_manager)

    def _run(self, query: str = "") -> str:
        """Fetch teams data"""
        teams = self.data_manager.get_teams_data()
        if teams:
            return f"Successfully fetched {len(teams)} teams with records, batting, pitching, and fielding statistics."
        return "Failed to fetch teams data. Please try again."

class AnalyzeTeamPerformanceTool(BaseTool):
    """Advanced tool for team performance analysis"""

    name: str = "analyze_team_performance"
    description: str = "Analyze a specific team's performance metrics and provide insights. Input should be a team code like 'HOU', 'NYY', etc."
    data_manager: BaseballDataManager

    def __init__(self, data_manager: BaseballDataManager):
        super().__init__(data_manager=data_manager)

    def _run(self, team_code: str) -> str:
        """Analyze specific team performance"""
        try:
            players = self.data_manager.get_players_data()
            teams = self.data_manager.get_teams_data()

            if not players or not teams:
                return "Unable to fetch data for team analysis."

            # Find team
            team_info = None
            for team in teams:
                if team.get('code') == team_code.upper():
                    team_info = team
                    break

            if not team_info:
                return f"Team {team_code} not found. Please use a valid team code like HOU, NYY, BOS, etc."

            # Get team players
            team_players = [p for p in players if p.get('team') == team_code.upper()]

            # Calculate team stats
            total_hits = sum(p.get('stats', {}).get('batting', {}).get('hits', 0) for p in team_players)
            total_hr = sum(p.get('stats', {}).get('batting', {}).get('homeRuns', 0) for p in team_players)
            total_rbi = sum(p.get('stats', {}).get('batting', {}).get('rbi', 0) for p in team_players)

            wins = team_info.get('wins', 0)
            losses = team_info.get('losses', 0)

            return f"""
Team {team_code.upper()} Analysis:
- Record: {wins}-{losses} ({wins/(wins+losses)*100:.1f}% win rate if wins+losses > 0 else 'No games played')
- Team Batting: {total_hits} hits, {total_hr} HR, {total_rbi} RBI
- Active Players: {len(team_players)}
- Performance Level: {'Strong' if wins > losses else 'Struggling' if wins+losses > 0 else 'Preseason'}
"""

        except Exception as e:
            return f"Error analyzing team {team_code}: {str(e)}"

class FilterPlayersByLastNameTool(BaseTool):
    """Tool for filtering players by last name"""

    name: str = "filter_players_by_last_name"
    description: str = "Filter players whose last names start with a specific letter. Input should be a single letter like 'A', 'T', etc."
    data_manager: BaseballDataManager

    def __init__(self, data_manager: BaseballDataManager):
        super().__init__(data_manager=data_manager)

    def _run(self, letter: str) -> str:
        """Filter players by last name"""
        try:
            players = self.data_manager.get_players_data()
            if not players:
                return "Unable to fetch players data."

            letter = letter.upper().strip()
            if len(letter) != 1 or not letter.isalpha():
                return "Please provide a single letter (A-Z) for filtering."

            filtered_players = []
            for player in players:
                if isinstance(player, dict):
                    name = player.get('name', 'Unknown')
                    if name != 'Unknown':
                        name_parts = name.strip().split()
                        if len(name_parts) >= 2:
                            # Handle suffixes
                            suffixes = ['jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'v']
                            last_part = name_parts[-1].lower()

                            if last_part in suffixes and len(name_parts) >= 3:
                                last_name = name_parts[-2]
                            else:
                                last_name = name_parts[-1]

                            last_name = re.sub(r'[^\w]', '', last_name)
                            if last_name.upper().startswith(letter):
                                filtered_players.append(player)

            if not filtered_players:
                return f"No players found with last names starting with '{letter}'."

            # Sort by hits
            sorted_players = []
            for player in filtered_players:
                if 'stats' in player and 'batting' in player['stats']:
                    hits = player['stats']['batting'].get('hits', 0)
                    sorted_players.append((player, hits))

            sorted_players.sort(key=lambda x: x[1], reverse=True)
            player_names = [p.get('name', 'Unknown') for p, hits in sorted_players[:20]]  # Top 20

            return f"Found {len(filtered_players)} players with last names starting with '{letter}': {', '.join(player_names)}"

        except Exception as e:
            return f"Error filtering players: {str(e)}"

class BuildBestTeamTool(BaseTool):
    """Tool for building the optimal baseball team from all available players"""

    name: str = "build_best_team"
    description: str = "Build the best possible baseball team by selecting optimal players for each position based on comprehensive statistics."
    data_manager: BaseballDataManager

    def __init__(self, data_manager: BaseballDataManager):
        super().__init__(data_manager=data_manager)

    def _run(self, query: str = "") -> str:
        """Build the best team from all available players"""
        try:
            players = self.data_manager.get_players_data()
            if not players:
                return "Unable to fetch players data for team building."

            # Define positions and their primary stats for ranking
            positions = {
                'C': {'primary': 'avg', 'secondary': 'rbi', 'reason': 'Catchers need good batting average for consistent offense and RBI production'},
                '1B': {'primary': 'rbi', 'secondary': 'avg', 'reason': 'First basemen are key RBI producers with good contact hitting'},
                '2B': {'primary': 'avg', 'secondary': 'runs', 'reason': 'Second basemen need high average and ability to score runs from leadoff spots'},
                '3B': {'primary': 'rbi', 'secondary': 'homeRuns', 'reason': 'Third basemen provide power hitting with RBI and home run production'},
                'SS': {'primary': 'avg', 'secondary': 'runs', 'reason': 'Shortstops need consistent hitting and speed to score runs'},
                'LF': {'primary': 'avg', 'secondary': 'homeRuns', 'reason': 'Left fielders combine contact hitting with power potential'},
                'CF': {'primary': 'avg', 'secondary': 'runs', 'reason': 'Center fielders need speed and consistent hitting to score runs'},
                'RF': {'primary': 'rbi', 'secondary': 'homeRuns', 'reason': 'Right fielders are expected to drive in runs and provide power'},
                'SP': {'primary': 'era', 'secondary': 'strikeouts', 'reverse': True, 'reason': 'Starting pitchers need low ERA (dominance) and high strikeouts (stuff)'},
            }

            best_team = {}
            used_players = set()
            selection_details = {}

            for position, criteria in positions.items():
                primary_stat = criteria['primary']
                secondary_stat = criteria['secondary']
                reverse_sort = criteria.get('reverse', False)

                # Filter players who can play this position
                position_players = []
                for player in players:
                    if player.get('name') in used_players:
                        continue

                    # Check if player actually plays this position
                    player_position = player.get('position', '').upper()
                    
                    # Position matching logic
                    position_match = False
                    if position == 'SP':
                        # Only select actual pitchers for starting pitcher
                        if player_position in ['P', 'SP', 'RP', 'LHP', 'RHP']:
                            position_match = True
                    elif position == 'C':
                        if player_position in ['C']:
                            position_match = True
                    elif position in ['1B', '2B', '3B', 'SS']:
                        if player_position in [position, 'IF', 'UTIL']:
                            position_match = True
                    elif position in ['LF', 'CF', 'RF']:
                        if player_position in [position, 'OF', 'UTIL']:
                            position_match = True
                    
                    if not position_match:
                        continue

                    player_stats = player.get('stats', {}).get('batting', {})
                    if position == 'SP':
                        player_stats = player.get('stats', {}).get('pitching', {})

                    if player_stats and primary_stat in player_stats:
                        primary_value = player_stats.get(primary_stat, 0)
                        secondary_value = player_stats.get(secondary_stat, 0)

                        # Convert to comparable format
                        if isinstance(primary_value, str):
                            try:
                                primary_value = float(primary_value)
                            except:
                                primary_value = 0

                        if isinstance(secondary_value, str):
                            try:
                                secondary_value = float(secondary_value)
                            except:
                                secondary_value = 0

                        # Only include players with meaningful stats
                        if position == 'SP':
                            if primary_value > 0 and secondary_value > 0:  # ERA and strikeouts
                                position_players.append({
                                    'player': player,
                                    'primary': primary_value,
                                    'secondary': secondary_value
                                })
                        else:
                            if primary_value > 0:  # Batting stats
                                position_players.append({
                                    'player': player,
                                    'primary': primary_value,
                                    'secondary': secondary_value
                                })

                if position_players:
                    # Sort by primary stat, then secondary stat
                    position_players.sort(key=lambda x: (x['primary'], x['secondary']), reverse=not reverse_sort)
                    best_player = position_players[0]['player']
                    best_team[position] = best_player
                    used_players.add(best_player.get('name'))
                    
                    # Store selection details for explanation
                    selection_details[position] = {
                        'total_candidates': len(position_players),
                        'selection_criteria': criteria,
                        'top_stats': position_players[0],
                        'runner_up': position_players[1] if len(position_players) > 1 else None
                    }

            # Format the response
            if best_team:
                response = "🏆 **BEST BASEBALL TEAM WITH EXPLANATIONS** 🏆\n\n"
                position_names = {
                    'C': 'Catcher', '1B': 'First Base', '2B': 'Second Base', '3B': 'Third Base',
                    'SS': 'Shortstop', 'LF': 'Left Field', 'CF': 'Center Field', 'RF': 'Right Field',
                    'SP': 'Starting Pitcher'
                }

                count = 0
                for position in ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'SP']:
                    if position in best_team:
                        count += 1
                        player = best_team[position]
                        name = player.get('name', 'Unknown')
                        team = player.get('team', 'Unknown')
                        player_position = player.get('position', 'Unknown')
                        
                        # Get selection details
                        details = selection_details.get(position, {})
                        criteria = details.get('selection_criteria', {})
                        total_candidates = details.get('total_candidates', 0)
                        top_stats = details.get('top_stats', {})
                        runner_up = details.get('runner_up', {})

                        # Get relevant stats
                        if position == 'SP':
                            stats = player.get('stats', {}).get('pitching', {})
                            stat_display = f"ERA: {stats.get('era', 'N/A')}, SO: {stats.get('strikeouts', 'N/A')}, W: {stats.get('wins', 'N/A')}"
                            primary_value = top_stats.get('primary', 'N/A')
                            secondary_value = top_stats.get('secondary', 'N/A')
                        else:
                            stats = player.get('stats', {}).get('batting', {})
                            stat_display = f"AVG: {stats.get('avg', 'N/A')}, HR: {stats.get('homeRuns', 'N/A')}, RBI: {stats.get('rbi', 'N/A')}"
                            primary_value = top_stats.get('primary', 'N/A')
                            secondary_value = top_stats.get('secondary', 'N/A')

                        response += f"{count}. **{position_names[position]}**: {name} ({team}) - Actual Position: {player_position}\n"
                        response += f"   └─ Stats: {stat_display}\n"
                        
                        # Add explanation
                        primary_stat = criteria.get('primary', 'unknown')
                        secondary_stat = criteria.get('secondary', 'unknown')
                        reason = criteria.get('reason', 'Statistical excellence at this position')
                        
                        response += f"   💡 **Why Selected**: {reason}\n"
                        response += f"   📊 **Selection Criteria**: Primary: {primary_stat.upper()} ({primary_value}), Secondary: {secondary_stat.upper()} ({secondary_value})\n"
                        response += f"   🏅 **Competition**: Chosen from {total_candidates} eligible players at this position\n"
                        
                        if runner_up:
                            runner_up_player = runner_up.get('player', {})
                            runner_up_name = runner_up_player.get('name', 'Unknown')
                            response += f"   🥈 **Runner-up**: {runner_up_name} ({primary_stat}: {runner_up.get('primary', 'N/A')}, {secondary_stat}: {runner_up.get('secondary', 'N/A')})\n"
                        
                        response += "\n"

                return response
            else:
                return "Unable to build a complete team from the available data."

        except Exception as e:
            return f"Error building best team: {str(e)}"

class BaseballQueryAnalyzer:
    """Advanced query analysis for better intent detection"""

    def __init__(self):
        self.team_codes = [
            'HOU', 'NYY', 'BOS', 'BAL', 'TOR', 'TB', 'CLE', 'DET', 'MIN', 'CWS', 'KC',
            'LAA', 'OAK', 'SEA', 'TEX', 'ATL', 'MIA', 'NYM', 'PHI', 'WSH', 'MIL',
            'CHC', 'STL', 'PIT', 'CIN', 'ARI', 'COL', 'LAD', 'SD', 'SF'
        ]

        self.stat_mappings = {
            'hits': 'hits', 'home runs': 'homeRuns', 'hr': 'homeRuns',
            'doubles': 'doubles', '2b': 'doubles', 'triples': 'triples', '3b': 'triples',
            'rbi': 'rbi', 'runs': 'runs', 'avg': 'avg', 'average': 'avg',
            'ops': 'ops', 'slg': 'slg', 'slugging': 'slg', 'obp': 'obp'
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the user's query and extract intent"""
        query_lower = query.lower()

        # Determine query type
        is_player_query = any(keyword in query_lower for keyword in [
            'player', 'hr', 'home run', 'batting', 'rbi', 'avg', 'era',
            'strikeout', 'pitcher', 'hitter', 'catcher', 'outfielder', 'infielder'
        ])

        is_team_query = any(keyword in query_lower for keyword in [
            'team', 'teams', 'standing', 'record', 'win', 'loss', 'division',
            'league', 'rank', 'top', 'best', 'worst', 'al', 'nl'
        ])

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

        # Extract last name filter
        last_name_filter = None
        name_match = re.search(r'last name starts with (?:an?\s+)?([A-Za-z])', query_lower)
        if name_match:
            last_name_filter = name_match.group(1).upper()

        # Detect team building queries
        is_team_building = any(phrase in query_lower for phrase in [
            'best team', 'best baseball team', 'optimal team', 'dream team',
            'all star team', 'every position', 'build a team', 'put together'
        ])

        return {
            'is_player_query': is_player_query,
            'is_team_query': is_team_query,
            'requested_team': requested_team,
            'requested_stat': requested_stat,
            'last_name_filter': last_name_filter,
            'is_team_building': is_team_building,
            'query_type': 'team_building' if is_team_building else
                         'team_specific' if requested_team and requested_stat else
                         'last_name_filter' if last_name_filter else
                         'general'
        }

class LangChainBaseballAgent:
    """LangChain-powered baseball agent"""

    def __init__(self):
        self.data_manager = BaseballDataManager()
        self.query_analyzer = BaseballQueryAnalyzer()

        # Initialize tools
        self.tools = [
            FetchPlayersTool(self.data_manager),
            FetchTeamsTool(self.data_manager),
            AnalyzeTeamPerformanceTool(self.data_manager),
            FilterPlayersByLastNameTool(self.data_manager),
            BuildBestTeamTool(self.data_manager)
        ]

        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-3.5-turbo",
            max_tokens=1000
        )

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    def process_query(self, query: str) -> str:
        """Process a baseball query using LangChain agent"""
        try:
            # Analyze the query first
            analysis = self.query_analyzer.analyze_query(query)

            # Customize prompt based on query type
            if analysis['last_name_filter']:
                custom_prompt = f"""
You are a baseball expert. The user asked for players whose last name starts with '{analysis['last_name_filter']}'.

Use the filter_players_by_last_name tool to get the exact list of players.
Provide ONLY the list of players found - no additional commentary unless the tool fails.

Query: {query}
"""
            elif analysis['query_type'] == 'team_building':
                custom_prompt = f"""
You are a baseball expert. The user wants you to build the best possible baseball team with players in every position.

Use the build_best_team tool to create an optimal team from all available players.
Select the best players for each position based on their statistics and performance.

Query: {query}
"""
            elif analysis['query_type'] == 'team_specific':
                custom_prompt = f"""
You are a baseball expert. The user asked about {analysis['requested_team']} players and {analysis['requested_stat']}.

Use the available tools to fetch and analyze the relevant data.
Focus on providing accurate statistics and insights.

Query: {query}
"""
            else:
                custom_prompt = f"""
You are a baseball expert. Answer the user's baseball question using the available tools.

Query: {query}
"""

            response = self.agent.run(custom_prompt)
            return response

        except Exception as e:
            return f"Error processing query: {str(e)}. Please try rephrasing your question."

# Hybrid approach: Use LangChain for complex queries, fallback to simple method
def answer_baseball_question_with_langchain(query: str) -> str:
    """
    Enhanced version that uses LangChain for complex queries but keeps simple logic for basic ones
    """
    print(f"🔍 Processing query with LangChain: {query}")

    # Quick analysis to decide approach
    query_lower = query.lower()
    is_complex = (
        'last name starts with' in query_lower or
        'analyze' in query_lower or
        'compare' in query_lower or
        'performance' in query_lower or
        len(query.split()) > 15  # Long, complex queries
    )

    if is_complex:
        print("🤖 Using LangChain agent for complex query")
        agent = LangChainBaseballAgent()
        return agent.process_query(query)
    else:
        print("⚡ Using simple method for basic query")
        # Import and use your original simple function
        from ..simple.app import answer_baseball_question as simple_answer
        return simple_answer(query)

# Streamlit UI with LangChain integration
def main():
    st.set_page_config(page_title="⚾ LangChain Baseball AI Agent", page_icon="⚾", layout="wide")

    st.title("⚾ LangChain Baseball AI Agent")
    st.markdown("*Enhanced baseball analytics with LangChain integration*")

    # Sidebar with capabilities
    with st.sidebar:
        st.header("🎯 Enhanced Capabilities")
        st.markdown("""
        **LangChain Features:**
        - ✅ Intelligent tool selection
        - ✅ Conversation memory
        - ✅ Complex query analysis
        - ✅ Multi-step reasoning

        **Available Tools:**
        - 🔧 Fetch Players Data
        - 🔧 Fetch Teams Data
        - 🔧 Analyze Team Performance
        - 🔧 Filter by Last Name

        **Smart Query Processing:**
        - 🎯 Automatic intent detection
        - 🎯 Tool selection based on query
        - 🎯 Context-aware responses
        """)

        st.header("📊 Data Sources")
        st.markdown("""
        - **Players API**: 1600+ MLB players
        - **Teams API**: 30 MLB teams
        - **Stats**: Complete batting/pitching data
        - **Caching**: 30-minute data freshness
        """)

    # Mode selection
    mode = st.radio("Choose processing mode:",
                   ["Simple (Fast)", "LangChain (Advanced)", "Auto (Smart)"])

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

        # Get AI response based on mode
        with st.chat_message("assistant"):
            with st.spinner(f"Processing with {mode}..."):
                if mode == "Simple (Fast)":
                    from ..simple.app import answer_baseball_question
                    response = answer_baseball_question(prompt)
                elif mode == "LangChain (Advanced)":
                    agent = LangChainBaseballAgent()
                    response = agent.process_query(prompt)
                else:  # Auto mode
                    response = answer_baseball_question_with_langchain(prompt)

            st.markdown(response)

        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

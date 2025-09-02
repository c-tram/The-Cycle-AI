"""
LangChain Integration for Baseball AI Agent
Optional advanced features that can be added to the current simple architecture
"""

from langchain.tools import BaseTool
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from typing import Dict, List, Optional, Any
import requests
import json
from dataclasses import dataclass

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

class FetchPlayersTool(BaseTool):
    """LangChain tool for fetching player data"""

    name: str = "fetch_players_data"
    description: str = "Fetch comprehensive player statistics from the MLB players API"

    def _run(self, query: str = "") -> str:
        """Fetch players data"""
        try:
            response = requests.get("https://thecycle.online/api/v2/players", timeout=10)
            if response.status_code == 200:
                data = response.json()
                players = data.get('players', [])
                return f"Successfully fetched {len(players)} players with complete statistics"
            else:
                return f"Failed to fetch players data: {response.status_code}"
        except Exception as e:
            return f"Error fetching players data: {str(e)}"

class FetchTeamsTool(BaseTool):
    """LangChain tool for fetching team data"""

    name: str = "fetch_teams_data"
    description: str = "Fetch team records and statistics from the MLB teams API"

    def _run(self, query: str = "") -> str:
        """Fetch teams data"""
        try:
            response = requests.get("https://thecycle.online/api/v2/teams", timeout=10)
            if response.status_code == 200:
                data = response.json()
                teams = data.get('teams', [])
                return f"Successfully fetched {len(teams)} teams with records and stats"
            else:
                return f"Failed to fetch teams data: {response.status_code}"
        except Exception as e:
            return f"Error fetching teams data: {str(e)}"

class AnalyzeTeamPerformanceTool(BaseTool):
    """Advanced tool for team performance analysis"""

    name: str = "analyze_team_performance"
    description: str = "Analyze team performance metrics and provide insights"

    def _run(self, team_code: str) -> str:
        """Analyze specific team performance"""
        try:
            # Fetch both players and teams data
            players_response = requests.get("https://thecycle.online/api/v2/players", timeout=10)
            teams_response = requests.get("https://thecycle.online/api/v2/teams", timeout=10)

            if players_response.status_code != 200 or teams_response.status_code != 200:
                return "Failed to fetch data for team analysis"

            players_data = players_response.json().get('players', [])
            teams_data = teams_response.json().get('teams', [])

            # Find team
            team_info = None
            for team in teams_data:
                if team.get('code') == team_code:
                    team_info = team
                    break

            if not team_info:
                return f"Team {team_code} not found"

            # Get team players
            team_players = [p for p in players_data if p.get('team') == team_code]

            # Calculate team stats
            total_hits = sum(p.get('stats', {}).get('batting', {}).get('hits', 0) for p in team_players)
            total_hr = sum(p.get('stats', {}).get('batting', {}).get('homeRuns', 0) for p in team_players)
            total_rbi = sum(p.get('stats', {}).get('batting', {}).get('rbi', 0) for p in team_players)

            wins = team_info.get('wins', 0)
            losses = team_info.get('losses', 0)

            return f"""
Team {team_code} Analysis:
- Record: {wins}-{losses} ({wins/(wins+losses)*100:.1f}% win rate)
- Team Batting: {total_hits} hits, {total_hr} HR, {total_rbi} RBI
- Key Players: {len(team_players)} active players
- Performance: {'Strong' if wins > losses else 'Struggling'}
"""

        except Exception as e:
            return f"Error analyzing team {team_code}: {str(e)}"

class ComparePlayersTool(BaseTool):
    """Tool for comparing player statistics"""

    name: str = "compare_players"
    description: str = "Compare statistics between two or more players"

    def _run(self, player_names: str) -> str:
        """Compare players"""
        try:
            response = requests.get("https://thecycle.online/api/v2/players", timeout=10)
            if response.status_code != 200:
                return "Failed to fetch players data"

            players_data = response.json().get('players', [])
            names = [name.strip() for name in player_names.split(',')]

            found_players = []
            for name in names:
                for player in players_data:
                    if name.lower() in player.get('name', '').lower():
                        found_players.append(player)
                        break

            if len(found_players) < 2:
                return "Need at least 2 players to compare"

            comparison = "Player Comparison:\n"
            for player in found_players:
                name = player.get('name', 'Unknown')
                team = player.get('team', 'Unknown')
                stats = player.get('stats', {}).get('batting', {})

                comparison += f"\n{name} ({team}):"
                comparison += f"\n  AVG: {stats.get('avg', 0):.3f}"
                comparison += f"\n  HR: {stats.get('homeRuns', 0)}"
                comparison += f"\n  RBI: {stats.get('rbi', 0)}"
                comparison += f"\n  OPS: {stats.get('ops', 0):.3f}"

            return comparison

        except Exception as e:
            return f"Error comparing players: {str(e)}"

class BaseballAnalyticsAgent:
    """Advanced LangChain agent for baseball analytics"""

    def __init__(self):
        # Initialize tools
        self.tools = [
            FetchPlayersTool(),
            FetchTeamsTool(),
            AnalyzeTeamPerformanceTool(),
            ComparePlayersTool()
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
            handle_parsing_errors=True
        )

    def analyze_query(self, query: str) -> str:
        """Process a baseball query using the LangChain agent"""
        try:
            # Custom prompt for baseball context
            baseball_prompt = f"""
You are an expert baseball analyst with access to comprehensive MLB data.
Use the available tools to gather and analyze baseball information.
Always provide data-driven insights and be specific about statistics.

User Query: {query}

Remember to:
1. Use tools to fetch relevant data
2. Compare and analyze statistics
3. Provide context and insights
4. Be accurate and specific

Answer:"""

            response = self.agent.run(baseball_prompt)
            return response

        except Exception as e:
            return f"Error processing query: {str(e)}"

# Example usage functions
def create_baseball_agent():
    """Create and return a configured baseball analytics agent"""
    return BaseballAnalyticsAgent()

def advanced_baseball_query(query: str) -> str:
    """Process a query using the advanced LangChain agent"""
    agent = create_baseball_agent()
    return agent.analyze_query(query)

# Multi-step analysis workflow
def comprehensive_team_analysis(team_code: str) -> Dict[str, Any]:
    """Perform comprehensive team analysis using multiple tools"""
    agent = create_baseball_agent()

    analysis = {
        "team_overview": agent.analyze_query(f"Give me an overview of {team_code} performance"),
        "key_players": agent.analyze_query(f"Who are the top performers on {team_code}"),
        "strengths_weaknesses": agent.analyze_query(f"What are {team_code}'s strengths and weaknesses"),
        "comparison": agent.analyze_query(f"How does {team_code} compare to other teams in their division")
    }

    return analysis

# Example of how to integrate with vector search (conceptual)
class BaseballVectorSearch:
    """Conceptual vector search for baseball data (requires FAISS or similar)"""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_player_documents(self, players: List[Dict]):
        """Add player data as searchable documents"""
        for player in players:
            doc = f"""
            Player: {player.get('name', 'Unknown')}
            Team: {player.get('team', 'Unknown')}
            Position: {player.get('position', 'Unknown')}
            Batting Stats: {json.dumps(player.get('stats', {}).get('batting', {}))}
            Pitching Stats: {json.dumps(player.get('stats', {}).get('pitching', {}))}
            """
            self.documents.append(doc)
            # In real implementation, you'd generate embeddings here

    def semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        """Perform semantic search on baseball data"""
        # This would use actual vector similarity in a real implementation
        results = []
        query_lower = query.lower()

        for doc in self.documents:
            if any(term in doc.lower() for term in query_lower.split()):
                results.append(doc)
                if len(results) >= top_k:
                    break

        return results

if __name__ == "__main__":
    # Example usage
    agent = create_baseball_agent()

    # Test queries
    queries = [
        "How is Aaron Judge performing this season?",
        "Compare Shohei Ohtani and Juan Soto",
        "Analyze the Houston Astros team performance",
        "Who leads the league in home runs?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print(f"Response: {agent.analyze_query(query)}")
        print("-" * 50)

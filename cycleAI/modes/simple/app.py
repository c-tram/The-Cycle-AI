import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
import sys

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.baseball_utils import BaseballDataManager, setup_streamlit_config

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def answer_baseball_question(query):
    """
    Simple function that answers baseball questions using only the two API endpoints
    """
    print(f"🔍 Processing query: {query}")
    
    # Use shared data manager
    data_manager = BaseballDataManager()
    
    # Determine what data we need
    is_player_query = any(keyword in query.lower() for keyword in [
        'player', 'hr', 'home run', 'batting', 'rbi', 'avg', 'era', 
        'strikeout', 'pitcher', 'hitter', 'catcher', 'outfielder', 'infielder'
    ])
    
    is_team_query = any(keyword in query.lower() for keyword in [
        'team', 'teams', 'standing', 'record', 'win', 'loss', 'division', 
        'league', 'rank', 'top', 'best', 'worst', 'al', 'nl', 'american', 'national'
    ])
    
    # If no specific type detected, fetch both
    if not is_player_query and not is_team_query:
        is_player_query = True
        is_team_query = True
    
    all_data = {}
    
    # Fetch players data
    if is_player_query:
        try:
            print("📊 Fetching players data...")
            response = requests.get("https://thecycle.online/api/v2/players", timeout=10)
            if response.status_code == 200:
                players_data = response.json()
                player_count = len(players_data.get('players', []))
                print(f"✅ Got {player_count} players from API")
                all_data['players'] = players_data['players']
            else:
                print(f"❌ Players API failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Players API error: {e}")
    
    # Fetch teams data
    if is_team_query:
        try:
            print("📊 Fetching teams data...")
            response = requests.get("https://thecycle.online/api/v2/teams", timeout=10)
            if response.status_code == 200:
                teams_data = response.json()
                team_count = len(teams_data.get('teams', []))
                print(f"✅ Got {team_count} teams from API")
                all_data['teams'] = teams_data['teams']
            else:
                print(f"❌ Teams API failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Teams API error: {e}")
    
    # If we have no data, return error
    if not all_data:
        return "Sorry, I couldn't fetch the baseball data right now. Please try again later."
    
    # Manual mapping of team codes to leagues (since API doesn't provide this)
    team_league_map = {
        'BAL': 'AL', 'BOS': 'AL', 'NYY': 'AL', 'TB': 'AL', 'TOR': 'AL',  # AL East
        'CLE': 'AL', 'DET': 'AL', 'MIN': 'AL', 'CWS': 'AL', 'KC': 'AL',   # AL Central  
        'LAA': 'AL', 'OAK': 'AL', 'SEA': 'AL', 'TEX': 'AL', 'HOU': 'AL', # AL West
        'ATL': 'NL', 'MIA': 'NL', 'NYM': 'NL', 'PHI': 'NL', 'WSH': 'NL', # NL East
        'CHC': 'NL', 'CIN': 'NL', 'MIL': 'NL', 'PIT': 'NL', 'STL': 'NL', # NL Central
        'ARI': 'NL', 'COL': 'NL', 'LAD': 'NL', 'SD': 'NL', 'SF': 'NL'    # NL West
    }
    
    # Convert data to readable text for the AI
    context_parts = []
    
    if 'players' in all_data:
        context_parts.append("PLAYERS DATA:")
        
        # Check for team-specific queries
        query_lower = query.lower()
        requested_team = None
        
        # Look for team abbreviations in the query
        team_codes = ['HOU', 'NYY', 'BOS', 'BAL', 'TOR', 'TB', 'CLE', 'DET', 'MIN', 'CWS', 'KC', 
                     'LAA', 'OAK', 'SEA', 'TEX', 'ATL', 'MIA', 'NYM', 'PHI', 'WSH', 'MIL', 
                     'CHC', 'STL', 'PIT', 'CIN', 'ARI', 'COL', 'LAD', 'SD', 'SF']
        
        for team_code in team_codes:
            if team_code.lower() in query_lower or f"{team_code.lower()} players" in query_lower:
                requested_team = team_code
                break
        
        # Check for specific stat requests
        stat_requests = {
            'hits': 'hits',
            'home runs': 'homeRuns', 'hr': 'homeRuns',
            'doubles': 'doubles', '2b': 'doubles',
            'triples': 'triples', '3b': 'triples',
            'rbi': 'rbi',
            'runs': 'runs',
            'avg': 'avg', 'average': 'avg', 'batting average': 'avg',
            'ops': 'ops',
            'slg': 'slg', 'slugging': 'slg',
            'obp': 'obp', 'on base': 'obp'
        }

        requested_stat = None
        for stat_name, stat_key in stat_requests.items():
            if stat_name in query_lower or f"most {stat_name}" in query_lower or f"top {stat_name}" in query_lower:
                requested_stat = stat_key
                break

        # Check for last name filtering (e.g., "players whose last name starts with A" or "last name starts with T")
        last_name_filter = None
        import re
        # More flexible regex to handle different phrasings
        name_match = re.search(r'last name starts with (?:an?\s+)?([A-Za-z])', query_lower)
        if name_match:
            last_name_filter = name_match.group(1).upper()
            print(f"🔍 Detected last name filter: {last_name_filter}")  # Debug print

        # Process players based on query
        if requested_team and requested_stat:
            # Team-specific stat query
            team_players = []
            for player in all_data['players']:
                if (isinstance(player, dict) and
                    player.get('team') == requested_team and
                    'stats' in player and
                    'batting' in player['stats']):

                    batting = player['stats']['batting']
                    stat_value = batting.get(requested_stat, 0)
                    if stat_value > 0:  # Only include players with the stat
                        team_players.append((player, stat_value))

            # Sort by the requested stat (highest first)
            team_players.sort(key=lambda x: x[1], reverse=True)

            for player, stat_value in team_players:
                name = player.get('name', 'Unknown')
                position = player.get('position', 'Unknown')
                context_parts.append(f"{name} ({requested_team}, {position}) | {stat_value} {requested_stat}")

        elif last_name_filter:
            # Filter by last name starting with specific letter
            filtered_players = []
            print(f"🔍 Filtering for last names starting with: {last_name_filter}")

            for player in all_data['players']:
                if isinstance(player, dict):
                    name = player.get('name', 'Unknown')
                    if name != 'Unknown':
                        # More robust last name extraction
                        name_parts = name.strip().split()

                        if len(name_parts) >= 2:
                            # Remove common suffixes
                            suffixes = ['jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'v']
                            last_part = name_parts[-1].lower()

                            # If the last part is a suffix, use the second-to-last part
                            if last_part in suffixes:
                                if len(name_parts) >= 3:
                                    last_name = name_parts[-2]
                                else:
                                    last_name = name_parts[-1]  # Fallback
                            else:
                                last_name = name_parts[-1]

                            # Clean up the last name (remove punctuation)
                            last_name = re.sub(r'[^\w]', '', last_name)

                            if last_name.upper().startswith(last_name_filter):
                                filtered_players.append(player)
                                print(f"✅ Found: {name} -> {last_name}")  # Debug print
                        elif len(name_parts) == 1:
                            # Single name (rare in baseball)
                            if name_parts[0].upper().startswith(last_name_filter):
                                filtered_players.append(player)

            print(f"📊 Found {len(filtered_players)} players with last name starting with {last_name_filter}")

            # Sort filtered players by a meaningful stat (hits by default)
            sorted_players = []
            for player in filtered_players:
                if 'stats' in player and 'batting' in player['stats']:
                    batting = player['stats']['batting']
                    hits = batting.get('hits', 0)
                    sorted_players.append((player, hits))

            # Sort by hits (highest first)
            sorted_players.sort(key=lambda x: x[1], reverse=True)

            context_parts.append(f"PLAYERS WITH LAST NAME STARTING WITH '{last_name_filter}':")
            player_names = []
            for player, hits in sorted_players:
                name = player.get('name', 'Unknown')
                if name != 'Unknown':
                    player_names.append(name)

            # Show just the names in a clean list
            if player_names:
                context_parts.append(f"Found {len(player_names)} players: {', '.join(player_names[:50])}")  # Limit to 50 for readability
                if len(player_names) > 50:
                    context_parts.append(f"... and {len(player_names) - 50} more players")
            else:
                context_parts.append("No players found with that last name starting letter.")

        else:
            # Default behavior: Sort players by key stats to ensure top performers are included
            players_to_include = []

            # Get all players with batting stats
            for player in all_data['players']:
                if isinstance(player, dict) and 'stats' in player and 'batting' in player['stats']:
                    batting = player['stats']['batting']
                    doubles = batting.get('doubles', 0) or 0
                    hr = batting.get('homeRuns', 0) or 0
                    rbi = batting.get('rbi', 0) or 0
                    hits = batting.get('hits', 0) or 0
                    avg = batting.get('avg', 0) or 0

                    # Ensure all values are numeric
                    try:
                        doubles = float(doubles) if doubles else 0
                        hr = float(hr) if hr else 0
                        rbi = float(rbi) if rbi else 0
                        hits = float(hits) if hits else 0
                        avg = float(avg) if avg else 0

                        # Calculate a score to prioritize players with good stats
                        stat_score = doubles + hr + (rbi / 10) + (hits / 5) + (avg * 100)
                        players_to_include.append((player, stat_score))
                    except (ValueError, TypeError):
                        # Skip players with invalid stat data
                        continue

            # Sort by stat score (highest first) and take top 100 players
            if players_to_include:
                players_to_include.sort(key=lambda x: x[1], reverse=True)
                top_players = [player for player, score in players_to_include[:100]]

                for player in top_players:
                    name = player.get('name', 'Unknown')
                    team = player.get('team', 'Unknown')
                    position = player.get('position', 'Unknown')

                    # Get comprehensive stats
                    stats_text = ""
                    if 'stats' in player:
                        stats = player['stats']
                        if 'batting' in stats:
                            batting = stats['batting']
                            # Get all available batting stats
                            games = batting.get('gamesPlayed', 0)
                            ab = batting.get('atBats', 0)
                            runs = batting.get('runs', 0)
                            hits = batting.get('hits', 0)
                            doubles = batting.get('doubles', 0)
                            triples = batting.get('triples', 0)
                            hr = batting.get('homeRuns', 0)
                            rbi = batting.get('rbi', 0)
                            sb = batting.get('stolenBases', 0)
                            cs = batting.get('caughtStealing', 0)
                            bb = batting.get('walks', 0)
                            so = batting.get('strikeOuts', 0)
                            hbp = batting.get('hitByPitch', 0)
                            sf = batting.get('sacrificeFlies', 0)
                            avg = batting.get('avg', 0)
                            obp = batting.get('obp', 0)
                            slg = batting.get('slg', 0)
                            ops = batting.get('ops', 0)
                            wrc_plus = batting.get('wrcPlus', 0)
                            war = batting.get('war', 0)

                            stats_text = f" | G:{games}, AB:{ab}, R:{runs}, H:{hits}, 2B:{doubles}, 3B:{triples}, HR:{hr}, RBI:{rbi}, SB:{sb}, BB:{bb}, SO:{so}, AVG:{avg}, OBP:{obp}, SLG:{slg}, OPS:{ops}"

                            if isinstance(wrc_plus, (int, float)) and wrc_plus > 0:
                                stats_text += f", wRC+:{wrc_plus}"
                            if isinstance(war, (int, float)) and war > 0:
                                stats_text += f", WAR:{war}"

                        if 'pitching' in stats:
                            pitching = stats['pitching']
                            # Get all available pitching stats
                            wins = pitching.get('wins', 0)
                            losses = pitching.get('losses', 0)
                            era = pitching.get('era', 0)
                            games = pitching.get('games', 0)
                            gs = pitching.get('gamesStarted', 0)
                            sv = pitching.get('saves', 0)
                            ip = pitching.get('inningsPitched', 0)
                            h = pitching.get('hits', 0)
                            r = pitching.get('runs', 0)
                            er = pitching.get('earnedRuns', 0)
                            hr = pitching.get('homeRuns', 0)
                            bb = pitching.get('walks', 0)
                            so = pitching.get('strikeOuts', 0)
                            whip = pitching.get('whip', 0)
                            k9 = pitching.get('kPer9', 0)
                            bb9 = pitching.get('bbPer9', 0)

                            if stats_text:
                                stats_text += f" | "
                            else:
                                stats_text = " | "

                            stats_text += f"W:{wins}, L:{losses}, ERA:{era}, G:{games}, GS:{gs}, SV:{sv}, IP:{ip}, H:{h}, ER:{er}, HR:{hr}, BB:{bb}, SO:{so}, WHIP:{whip}, K/9:{k9}, BB/9:{bb9}"

                    context_parts.append(f"- {name} ({team}, {position}){stats_text}")
        
    if 'teams' in all_data:
        context_parts.append("\nTEAMS DATA:")
        for team in all_data['teams']:  # Get all teams
            if isinstance(team, dict):
                name = team.get('name', 'Unknown')
                city = team.get('city', 'Unknown')
                division = team.get('division', 'Unknown')
                
                # Use manual league mapping since API doesn't provide this
                league = team_league_map.get(name, team.get('league', 'Unknown'))

                # Get comprehensive team stats
                record_text = ""
                if 'record' in team:
                    record = team['record']
                    wins = record.get('wins', 0)
                    losses = record.get('losses', 0)
                    pct = record.get('pct', 0)
                    gb = record.get('gamesBack', 0)
                    if wins > 0 or losses > 0:
                        record_text = f" | Record: {wins}-{losses}"
                        if pct > 0:
                            record_text += f" ({pct:.3f})"
                        if gb > 0:
                            record_text += f", {gb} GB"

                # Get team stats
                stats_text = ""
                if 'stats' in team:
                    stats = team['stats']
                    
                    # Overall stats
                    if 'overall' in stats:
                        overall = stats['overall']
                        rs = overall.get('runsScored', 0)
                        ra = overall.get('runsAllowed', 0)
                        if rs > 0 or ra > 0:
                            stats_text += f" | RS: {rs}, RA: {ra}"
                    
                    # Batting stats
                    if 'batting' in stats:
                        batting = stats['batting']
                        ba = batting.get('avg', 0)
                        obp = batting.get('obp', 0)
                        slg = batting.get('slg', 0)
                        ops = batting.get('ops', 0)
                        hr = batting.get('homeRuns', 0)
                        rbi = batting.get('rbi', 0)
                        if ba > 0 or obp > 0 or slg > 0:
                            stats_text += f" | BA: {ba}, OBP: {obp}, SLG: {slg}, OPS: {ops}, HR: {hr}, RBI: {rbi}"
                    
                    # Pitching stats
                    if 'pitching' in stats:
                        pitching = stats['pitching']
                        era = pitching.get('era', 0)
                        whip = pitching.get('whip', 0)
                        so = pitching.get('strikeOuts', 0)
                        if era > 0 or whip > 0 or so > 0:
                            stats_text += f" | ERA: {era}, WHIP: {whip}, SO: {so}"
                    
                    # Fielding stats
                    if 'fielding' in stats:
                        fielding = stats['fielding']
                        fpct = fielding.get('fieldingPct', 0)
                        if fpct > 0:
                            stats_text += f" | FPCT: {fpct}"

                # Get CVR and WAR if available
                advanced_text = ""
                if 'cvr' in team:
                    cvr_value = team['cvr']
                    if isinstance(cvr_value, (int, float)) and cvr_value > 0:
                        advanced_text += f" | CVR: {cvr_value}"
                if 'war' in team:
                    war_value = team['war']
                    if isinstance(war_value, (int, float)) and war_value > 0:
                        advanced_text += f" | WAR: {war_value}"

                context_parts.append(f"- {city} {name} ({division}, {league}){record_text}{stats_text}{advanced_text}")

    context = "\n".join(context_parts)

    # Use OpenAI to answer based on the context
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Customize prompt based on query type
        if last_name_filter:
            prompt = f"""You are a baseball expert. The user asked for players whose last name starts with '{last_name_filter}'.

Use ONLY the player list provided below. Do NOT add extra players or information not in the data.
Just provide the list of players as shown in the data.

QUESTION: {query}

DATA:
{context}

Answer:"""
        else:
            prompt = f"""You are a baseball expert. Answer the following question using ONLY the data provided below.
If the data doesn't contain the information needed to fully answer the question, say so clearly.

QUESTION: {query}

DATA:
{context}

Answer:"""

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return f"I found the data but couldn't process it properly. Error: {e}"

# Streamlit UI
st.title("Baseball AI Agent - API Only")
st.markdown("This agent answers questions using only data from thecycle.online APIs")

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# Chat input
if prompt := st.chat_input("Ask a baseball question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Fetching data from APIs..."):
            try:
                answer = answer_baseball_question(prompt)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar
st.sidebar.title("🤖 AI Baseball Agent")
st.sidebar.markdown("### Data Sources")
st.sidebar.markdown("This agent uses only these two APIs:")
st.sidebar.markdown("- **Players API**: https://thecycle.online/api/v2/players")
st.sidebar.markdown("- **Teams API**: https://thecycle.online/api/v2/teams")

st.sidebar.markdown("### Current Capabilities")
st.sidebar.markdown("✅ **Real-time Data**: Fetches live baseball statistics")
st.sidebar.markdown("✅ **Player Stats**: Batting, pitching, and fielding data")
st.sidebar.markdown("✅ **Team Records**: Standings, wins, losses, and rankings")
st.sidebar.markdown("✅ **League Queries**: AL vs NL team comparisons")
st.sidebar.markdown("✅ **Smart Queries**: Automatically detects player vs team questions")

st.sidebar.markdown("### Data Freshness")
st.sidebar.markdown("📊 **Players**: Top 100 players by performance loaded")
st.sidebar.markdown("🏆 **Teams**: 30 MLB teams with AL/NL league data")
st.sidebar.markdown("🔄 **Updates**: Data refreshed on each query")

st.sidebar.markdown("### Answer Confidence")
st.sidebar.markdown("🎯 **High Confidence**: Direct stat lookups")
st.sidebar.markdown("🤔 **Medium Confidence**: Calculated rankings")
st.sidebar.markdown("⚠️ **Low Confidence**: Limited data available")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Recent Activity")
if 'messages' in st.session_state and len(st.session_state.messages) > 0:
    recent_queries = [msg['content'] for msg in st.session_state.messages[-4:] if msg['role'] == 'user']
    if recent_queries:
        st.sidebar.markdown("**Recent Questions:**")
        for i, query in enumerate(recent_queries[-3:], 1):
            st.sidebar.markdown(f"{i}. {query[:30]}{'...' if len(query) > 30 else ''}")

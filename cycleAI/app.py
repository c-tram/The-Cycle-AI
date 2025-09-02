"""
Intelligent Baseball AI Agent - Main Application
===============================================

This is the main Streamlit application that brings together all the intelligent components.
"""

import streamlit as st
import os
import time
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv
import pandas as pd
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

# Import organized components
from src.intelligence import IntelligentAgent, replace_parse_query_intent
from src.scrapers import SyncJavaScriptScraper, SyncQueryDrivenScraper

# Define the state for LangGraph
class AgentState(TypedDict):
    query: str
    enhanced_query: str
    search_results: List[dict]
    fetched_docs: List[str]
    context: str
    answer: str
    ratings: List[dict]
    min_games: Optional[int]

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize intelligent components
@st.cache_resource
def get_intelligent_agent():
    """Initialize and cache the intelligent agent"""
    return IntelligentAgent()

@st.cache_resource
def get_js_scraper():
    """Initialize and cache the JavaScript scraper"""
    return SyncJavaScriptScraper()

@st.cache_resource
def get_query_scraper():
    """Initialize and cache the query-driven scraper"""
    return SyncQueryDrivenScraper()

# Initialize components
intelligent_agent = get_intelligent_agent()
js_scraper = get_js_scraper()
query_scraper = get_query_scraper()

# Legacy learning system (keeping for backwards compatibility)
class QueryLearner:
    """Legacy learning system - now replaced by intelligent components"""
    def __init__(self):
        self.query_patterns = {}
        self.intent_accuracy = {}
        self.feedback_history = []

    def learn_from_feedback(self, query, intent, rating, response_quality):
        """Learn from user feedback to improve future query processing"""
        if not query:
            return

        query_key = self._normalize_query(query)

        # Store feedback
        feedback_entry = {
            "query": query,
            "normalized_query": query_key,
            "intent": intent,
            "rating": rating,
            "quality": response_quality,
            "timestamp": time.time()
        }
        self.feedback_history.append(feedback_entry)

        # Update pattern recognition
        if query_key not in self.query_patterns:
            self.query_patterns[query_key] = {
                "successful_intent": intent,
                "success_count": 0,
                "total_count": 0,
                "avg_rating": 0
            }

        pattern = self.query_patterns[query_key]
        pattern["total_count"] += 1

        if rating == "good":
            pattern["success_count"] += 1
            pattern["avg_rating"] = (pattern["avg_rating"] * (pattern["total_count"] - 1) + 1) / pattern["total_count"]
        else:
            pattern["avg_rating"] = (pattern["avg_rating"] * (pattern["total_count"] - 1) + 0) / pattern["total_count"]

        # Update intent accuracy
        intent_key = f"{intent['type']}_{intent.get('stat', 'unknown')}"
        if intent_key not in self.intent_accuracy:
            self.intent_accuracy[intent_key] = {"correct": 0, "total": 0}

        self.intent_accuracy[intent_key]["total"] += 1
        if rating == "good":
            self.intent_accuracy[intent_key]["correct"] += 1

    def get_improved_intent(self, query):
        """Get improved intent based on learning history"""
        query_key = self._normalize_query(query)

        if query_key in self.query_patterns:
            pattern = self.query_patterns[query_key]
            if pattern["success_count"] / pattern["total_count"] > 0.7:  # 70% success rate
                return pattern["successful_intent"]

        return None  # Use default intent parsing

    def get_best_data_source(self, intent):
        """Recommend best data source based on historical success"""
        intent_key = f"{intent['type']}_{intent.get('stat', 'unknown')}"

        if intent_key in self.intent_accuracy:
            accuracy = self.intent_accuracy[intent_key]["correct"] / self.intent_accuracy[intent_key]["total"]
            if accuracy > 0.8:
                return "learned_preference"

        return "default"

    def _normalize_query(self, query):
        """Normalize query for pattern matching"""
        if not query:
            return ""

        import re
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        # Simple stemming - remove common endings
        words = normalized.split()
        stemmed_words = []
        for word in words:
            if word.endswith('ing'):
                word = word[:-3]
            elif word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('s') and len(word) > 1:
                word = word[:-1]
            stemmed_words.append(word)

        return ' '.join(stemmed_words)

    def get_learning_stats(self):
        """Get statistics about learning progress"""
        total_feedback = len(self.feedback_history)
        good_ratings = sum(1 for f in self.feedback_history if f["rating"] == "good")
        accuracy_rate = good_ratings / total_feedback if total_feedback > 0 else 0

        return {
            "total_feedback": total_feedback,
            "accuracy_rate": accuracy_rate,
            "learned_patterns": len(self.query_patterns),
            "intent_types_learned": len(self.intent_accuracy)
        }

# Initialize legacy learner for backwards compatibility
query_learner = QueryLearner()

# Load learning data on startup
try:
    with open('data/learning_data.json', 'r') as f:
        learning_data = json.load(f)
        query_learner.query_patterns = learning_data.get("query_patterns", {})
        query_learner.intent_accuracy = learning_data.get("intent_accuracy", {})
        query_learner.feedback_history = learning_data.get("feedback_history", [])
except:
    pass  # No existing learning data

def parse_query_intent(query, conversation_history=None):
    """
    INTELLIGENT VERSION: Uses semantic understanding, learning, and contextual reasoning
    instead of rule-based pattern matching.
    """
    # Use the intelligent agent instead of rules
    return replace_parse_query_intent(intelligent_agent, query, conversation_history)

def parse_and_filter_tables(html_content, query, intent=None, scrape_data=None):
    """Parse tables with intelligent filtering and JavaScript scraping support"""
    try:
        from io import StringIO
        import numpy as np

        # 🧠 INTELLIGENT DATA PROCESSING
        # First try to use structured data from JavaScript scraping
        if scrape_data and scrape_data.get("tables"):
            print("🚀 Using structured data from JavaScript scraper")
            tables = []
            for table_info in scrape_data["tables"]:
                # Convert back to pandas DataFrame
                df = pd.DataFrame(table_info["data"])
                tables.append(df)
            filtered_text = "Data extracted from JavaScript-rendered page:\n\n"
        else:
            # Fallback to HTML parsing
            print(f"DEBUG: HTML content length: {len(html_content)} characters")
            if "<table" in html_content.lower():
                print("DEBUG: Found <table> tags in HTML")
                table_count = html_content.lower().count("<table")
                print(f"DEBUG: Number of <table> tags: {table_count}")
            else:
                print("DEBUG: No <table> tags found in HTML")
                # Check for other data structures
                soup = BeautifulSoup(html_content, 'html')
                divs_with_data = soup.find_all('div', class_=lambda x: x and any(word in x.lower() for word in ['player', 'stat', 'data']))
                print(f"DEBUG: Found {len(divs_with_data)} divs with potential data")

            tables = pd.read_html(StringIO(html_content))
            filtered_text = ""

        # Debug: Log what tables we found
        print(f"DEBUG: Found {len(tables)} tables total")
        for i, table in enumerate(tables):
            print(f"DEBUG: Table {i} columns: {list(table.columns)}")
            print(f"DEBUG: Table {i} shape: {table.shape}")

        if not tables:
            print("DEBUG: No tables found, using fallback extraction")
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()
            return text_content[:2000] if len(text_content) > 100 else "No readable data found."

        # INTELLIGENT TABLE PROCESSING: Use execution plan if available
        execution_plan = intent.get("execution_plan") if intent and isinstance(intent, dict) else None
        filters = execution_plan.get("table_processing", {}).get("filters", []) if execution_plan else []

        query_lower = query.lower() if query else ""

        for i, table in enumerate(tables):
            filtered_text += f"\n--- Table {i+1} ---\n"

            # Apply intelligent filters
            for filter_spec in filters:
                column_concept = filter_spec.get("column_concept", "")
                operator = filter_spec.get("operator", ">=")
                value = filter_spec.get("value", 0)

                print(f"DEBUG: Applying filter {column_concept} {operator} {value}")

                # Find matching columns (fuzzy match)
                matching_cols = []
                for col in table.columns:
                    col_lower = str(col).lower()
                    if any(keyword in col_lower for keyword in ["game", "g", "gp"]) and "games" in column_concept:
                        matching_cols.append(col)

                for col in matching_cols:
                    print(f"DEBUG: Filtering by column '{col}' {operator} {value}")
                    table[col] = pd.to_numeric(table[col], errors='coerce')
                    if operator == ">=":
                        table = table[table[col] >= value]
                    filtered_text += f"Filtered to players with {col} {operator} {value}:\n"
                    break

            # Enhanced target stat detection
            target_stat = intent.get("target_stat", "") if intent else ""

            # Sort by target stat if found
            if target_stat:
                stat_cols = [col for col in table.columns if target_stat.lower() in col.lower() or col.lower() in target_stat.lower()]
                if stat_cols:
                    stat_col = stat_cols[0]
                    print(f"DEBUG: Sorting table {i} by {stat_col}")
                    table[stat_col] = pd.to_numeric(table[stat_col], errors='coerce')
                    table = table.sort_values(stat_col, ascending=False, na_last=True)
                    filtered_text += f"Sorted by {stat_col} (highest first):\n"
                else:
                    print(f"DEBUG: Target stat '{target_stat}' not found in columns: {list(table.columns)}")
                    # Try fuzzy matching for common stats
                    for col in table.columns:
                        if "cvr" in col.lower() and "cvr" in target_stat.lower():
                            table[col] = pd.to_numeric(table[col], errors='coerce')
                            table = table.sort_values(col, ascending=False, na_last=True)
                            filtered_text += f"Sorted by {col} (highest first):\n"
                            break

            # If no specific filtering, try to find relevant stats
            elif any(word in query_lower for word in ["highest", "most", "best", "top", "leader"]):
                # Look for numeric columns that might be stats
                numeric_cols = table.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # Sort by first numeric column (likely a key stat)
                    sort_col = numeric_cols[0]
                    table = table.sort_values(sort_col, ascending=False)
                    filtered_text += f"Sorted by {sort_col} (highest first):\n"

            filtered_text += table.to_string(index=False)
            filtered_text += "\n"
        return filtered_text
    except Exception as e:
        print(f"DEBUG: Error parsing tables: {e}")
        # Enhanced fallback - try to extract any data structure
        soup = BeautifulSoup(html_content, 'html.parser')

        # Look for data in divs, spans, or other structures
        data_text = soup.get_text()
        if len(data_text) > 100:
            return data_text[:2000]  # Return first 2000 chars
        else:
            return "No readable data found in the webpage."

# Node functions for LangGraph
def search_node(state: AgentState) -> AgentState:
    query = state.get("query", "")
    if not query:
        return state

    # 🧠 INTELLIGENT PROCESSING: Use the intelligent agent instead of rules
    conversation_history = state.get("conversation_history", [])
    intelligent_query = intelligent_agent.process_query(query, conversation_history)

    # Store intelligent analysis in state for display
    if 'current_intelligent_query' not in st.session_state:
        st.session_state.current_intelligent_query = None
    st.session_state.current_intelligent_query = intelligent_query

    enhanced_query = query
    if "this year" in query.lower() or "current" in query.lower() or "2025" in query.lower():
        enhanced_query += " 2025 MLB news"
    else:
        enhanced_query += " MLB"

    # Use intelligent execution plan instead of rule-based logic
    execution_plan = intelligent_query.execution_plan
    data_source = execution_plan.get("data_source", "https://thecycle.online/players")
    endpoint = execution_plan.get("endpoint", "players")

    # Store analysis for later display
    confidence = intelligent_query.confidence

    try:
        print(f"🧠 INTELLIGENT SCRAPING: Using endpoint '{endpoint}' for query analysis...")
        scrape_result = js_scraper.scrape_players(wait_time=3, endpoint=endpoint)

        if scrape_result.get("data_found"):
            print(f"✅ Successfully scraped with method: {scrape_result['extraction_method']}")
            results = [{
                "html": scrape_result["html_content"],
                "source": data_source,
                "intelligent_query": intelligent_query,  # Pass intelligent analysis forward
                "confidence": confidence,
                "execution_plan": execution_plan,
                "scrape_data": scrape_result  # Include structured data
            }]
        else:
            print("❌ JavaScript scraping failed, falling back to basic requests")
            # Fallback to basic request
            response = requests.get(data_source, timeout=10)
            if response.status_code == 200:
                results = [{
                    "html": response.text,
                    "source": data_source,
                    "intelligent_query": intelligent_query,
                    "confidence": confidence,
                    "execution_plan": execution_plan,
                    "scrape_data": {"error": "Fallback to basic scraping"}
                }]
            else:
                results = []
    except Exception as e:
        print(f"❌ Error with intelligent scraping: {e}")
        # Ultimate fallback
        try:
            response = requests.get(data_source, timeout=10)
            if response.status_code == 200:
                results = [{
                    "html": response.text,
                    "source": data_source,
                    "error": f"Scraping failed: {e}"
                }]
            else:
                results = []
        except:
            results = []

    return {**state, "enhanced_query": enhanced_query, "search_results": results}

def fetch_node(state: AgentState) -> AgentState:
    results = state["search_results"]
    docs = []
    for result in results:
        if "html" in result:
            # Parse tables from HTML content with intelligent scraping support
            html_content = result["html"]
            intent = result.get("intent")
            scrape_data = result.get("scrape_data")  # Get structured data from JS scraping

            parsed_text = parse_and_filter_tables(
                html_content,
                state.get("query", ""),
                intent,
                scrape_data
            )
            docs.append(parsed_text)

    return {**state, "fetched_docs": docs}

def index_node(state: AgentState) -> AgentState:
    docs = state["fetched_docs"]
    if not docs:
        return state

    # Initialize vectorstore if not exists
    if 'vectorstore' not in st.session_state:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        st.session_state.vectorstore = Chroma(embedding_function=embeddings, persist_directory="./data/chroma_db")

    # Split and index documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = []
    for doc in docs:
        splits = text_splitter.split_text(doc)
        all_splits.extend(splits)

    if all_splits:
        st.session_state.vectorstore.add_texts(all_splits)

    return state

def retrieve_node(state: AgentState) -> AgentState:
    query = state.get("query", "")
    if not query or 'vectorstore' not in st.session_state:
        return state

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs if doc.page_content])
    return {**state, "context": context}

def answer_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    context = state.get("context", "")
    query = state.get("query", "")
    search_results = state.get("search_results", [])

    if not query:
        answer = "Please provide a question."
        return {**state, "answer": answer}

    # Get intelligent analysis from search results
    intelligent_query = None
    execution_plan = None
    confidence = 0.5

    if search_results:
        for result in search_results:
            if "intelligent_query" in result:
                intelligent_query = result["intelligent_query"]
                execution_plan = result.get("execution_plan", {})
                confidence = result.get("confidence", 0.5)
                break

    # Build enhanced prompt with intelligent insights
    prompt_parts = []

    if context:
        prompt_parts.append(f"Based on the following data: {context}")
    else:
        prompt_parts.append("Using your knowledge of baseball statistics and data")

    # Add intelligent reasoning context
    if intelligent_query and execution_plan:
        reasoning_chain = execution_plan.get("reasoning_chain", [])
        if reasoning_chain:
            prompt_parts.append(f"The AI system analyzed this query with the following reasoning:")
            for step in reasoning_chain:
                prompt_parts.append(f"- {step}")
            prompt_parts.append(f"System confidence: {confidence:.1%}")

        # Add response style guidance
        response_strategy = execution_plan.get("response_strategy", {})
        answer_style = response_strategy.get("style", "informative")

        if answer_style == "actionable":
            prompt_parts.append("Provide actionable advice suitable for fantasy baseball decisions.")
        elif answer_style == "analytical":
            prompt_parts.append("Provide detailed analytical insights with context.")
        elif answer_style == "comparative":
            prompt_parts.append("Focus on clear comparisons between players or stats.")
        elif answer_style == "direct":
            prompt_parts.append("Be direct and concise due to urgency.")

        # Add confidence communication if needed
        if response_strategy.get("include_confidence"):
            prompt_parts.append("Include notes about data reliability and sample sizes.")

    prompt_parts.append(f"\nQuestion: {query}\n\nAnswer:")
    final_prompt = "\n".join(prompt_parts)

    try:
        answer = llm.invoke(final_prompt).content

        # Add intelligent insights to the answer
        if intelligent_query and confidence > 0.8:
            answer += f"\n\n*AI Confidence: {confidence:.1%} - High confidence in analysis*"
        elif confidence < 0.5:
            answer += f"\n\n*AI Confidence: {confidence:.1%} - Recommend verifying results*"

    except Exception as e:
        answer = f"Error generating answer: {e}"

    return {**state, "answer": answer}

def process_user_feedback(query, intent, rating, response_quality="unknown", data_found=None):
    """Process user feedback to improve the INTELLIGENT learning system"""
    # Old system learning (for backwards compatibility)
    query_learner.learn_from_feedback(query, intent, rating, response_quality)

    # NEW: Intelligent agent learning
    if data_found is None:
        data_found = {"tables": []}  # Default empty data structure

    intelligent_agent.learn_from_feedback(query, data_found, rating)

    # Save learning data (optional - for persistence)
    try:
        with open('data/learning_data.json', 'w') as f:
            json.dump({
                "query_patterns": query_learner.query_patterns,
                "intent_accuracy": query_learner.intent_accuracy,
                "feedback_history": query_learner.feedback_history[-100:]  # Keep last 100 entries
            }, f, indent=2)
    except:
        pass  # Silently fail if can't save

# Load environment variables
load_dotenv()

# Set OpenAI API key from .env
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize session state for persistence
if 'vectorstore' not in st.session_state:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    st.session_state.vectorstore = Chroma(embedding_function=embeddings, persist_directory="./data/chroma_db")

if 'agent_graph' not in st.session_state:
    # Build the LangGraph
    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("fetch", fetch_node)
    workflow.add_node("index", index_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)

    # Define edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "fetch")
    workflow.add_edge("fetch", "index")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)

    st.session_state.agent_graph = workflow.compile()

# Streamlit UI
st.title("🧠 Intelligent Baseball AI Agent")
st.markdown("Ask questions about baseball players, teams, and statistics!")

# Chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# Chat input
if prompt := st.chat_input("Ask a baseball question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.current_prompt = prompt  # Store for rating buttons
    with st.chat_message("user"):
        st.write(prompt)

with st.chat_message("assistant"):
    with st.spinner("Researching..."):
        try:
            # Use LangGraph
            initial_state = {
                "query": prompt,
                "enhanced_query": "",
                "search_results": [],
                "fetched_docs": [],
                "context": "",
                "answer": "",
                "ratings": st.session_state.ratings if 'ratings' in st.session_state else [],
                "min_games": None
            }
            result = st.session_state.agent_graph.invoke(initial_state)
            answer = result["answer"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # 🧠 INTELLIGENT REASONING DISPLAY
            if st.session_state.get('current_intelligent_query'):
                with st.expander("🧠 AI Reasoning (Click to see how I understood your question)", expanded=False):
                    iq = st.session_state.current_intelligent_query
                    reasoning = intelligent_agent.explain_reasoning(iq)
                    st.markdown(reasoning)

                    # Show confidence with color
                    confidence = iq.confidence
                    if confidence > 0.8:
                        st.success(f"🎯 High Confidence: {confidence:.1%}")
                    elif confidence > 0.6:
                        st.info(f"🤔 Medium Confidence: {confidence:.1%}")
                    else:
                        st.warning(f"⚠️ Low Confidence: {confidence:.1%} - Results may be unreliable")

            # Store intent for learning
            current_prompt = st.session_state.get("current_prompt", "")
            current_intent = result.get("intent", parse_query_intent(current_prompt) if current_prompt else {"type": "unknown"})

            # Rating buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Good", key=f"good_{len(st.session_state.messages)-1}"):
                    st.session_state.ratings.append({"question": current_prompt, "answer": answer, "rating": "good"})
                    with open('data/ratings.json', 'w') as f:
                        json.dump(st.session_state.ratings, f)
                    # Learn from positive feedback
                    if current_prompt:
                        process_user_feedback(current_prompt, current_intent, "good", "high")
                    st.success("Rated as good! AI will learn from this.")
            with col2:
                if st.button("👎 Bad", key=f"bad_{len(st.session_state.messages)-1}"):
                    st.session_state.ratings.append({"question": current_prompt, "answer": answer, "rating": "bad"})
                    with open('data/ratings.json', 'w') as f:
                        json.dump(st.session_state.ratings, f)
                    # Learn from negative feedback
                    if current_prompt:
                        process_user_feedback(current_prompt, current_intent, "bad", "low")
                    st.success("Rated as bad. AI will improve from this feedback.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})

# Sidebar with information
st.sidebar.title("🤖 AI Intelligence")
st.sidebar.markdown("### Current Capabilities")
st.sidebar.markdown("- ✅ Semantic understanding of baseball concepts")
st.sidebar.markdown("- ✅ Learning from user feedback")
st.sidebar.markdown("- ✅ JavaScript website scraping")
st.sidebar.markdown("- ✅ Transparent reasoning display")
st.sidebar.markdown("- ✅ Confidence scoring")

st.sidebar.markdown("### Data Sources")
st.sidebar.markdown("- thecycle.online (primary)")
st.sidebar.markdown("- DuckDuckGo search (fallback)")
st.sidebar.markdown("- Local knowledge base")

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.rerun()

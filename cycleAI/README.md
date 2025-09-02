# 🧠 Intelligent Baseball AI Agent

**A truly intelligent baseball analytics system that learns, reasons, and adapts - not just pattern matching.**

## 🎯 What This System Does

This is **not** a simple chatbot. This is an intelligent agent that:

- **Understands baseball concepts** (CVR, OPS, WAR, etc.) and their relationships
- **Learns from user feedback** to improve over time
- **Reasons about user intent** - why are you asking this question?
- **Scrapes modern JavaScript websites** (thecycle.online) with full browser automation
- **Provides transparent reasoning** - you can see exactly how it thinks
- **Adapts to your needs** through continuous learning

## 🏗️ Current Architecture

### Core Intelligence Components

1. **🧠 Intelligent Agent** (`intelligent_agent.py`)
   - Coordinates semantic understanding, learning, and reasoning
   - Replaces rule-based pattern matching with true intelligence
   - Provides confidence scores and reasoning explanations

2. **📚 Semantic Engine** (`semantic_engine.py`)
   - Understands baseball concepts and their relationships
   - Extracts constraints (minimum games, team filters, etc.)
   - Maps natural language to structured queries

3. **🎓 Learning System** (`learning_system.py`)
   - Learns from user feedback to improve future responses
   - Remembers successful query patterns
   - Adapts column mappings and data sources

4. **🤔 Contextual Reasoner** (`contextual_reasoner.py`)
   - Understands WHY users ask questions (scouting, fantasy, research)
   - Assesses urgency and depth requirements
   - Generates appropriate response strategies

5. **🌐 JavaScript Scraper** (`js_scraper.py`)
   - Uses Playwright to render modern JavaScript websites
   - Extracts structured data from dynamic content
   - Handles interactive elements and filters

6. **🎯 Query-Driven Scraper** (`query_driven_scraper.py`)
   - Analyzes queries to determine needed filters
   - Interacts with website UI to get exactly the data you need
   - Handles team filters, stat selection, sorting

### Data Flow

```
User Query → Intelligent Agent → Semantic Understanding → Learning Insights → Contextual Reasoning → Execution Plan → JavaScript Scraper → Data Analysis → Answer with Reasoning
```

## 🚀 Key Features

### Intelligence Features
- **Semantic Understanding**: Knows that "highest CVR with 80+ games" means filter by games ≥ 80, sort by CVR descending
- **Learning**: Remembers which approaches work and adapts
- **Context Awareness**: Knows if you're scouting, playing fantasy, or researching
- **Transparent Reasoning**: Shows you exactly how it arrived at its answer

### Technical Features
- **Modern Web Scraping**: Handles JavaScript-rendered sites with Playwright
- **LangGraph Workflows**: Advanced state management for complex queries
- **Vector Search**: ChromaDB for document retrieval
- **Feedback Learning**: Improves based on user ratings
- **Streamlit UI**: Clean web interface with reasoning display

## 📊 Current Capabilities

### What It Can Do Well
- ✅ Understand complex baseball queries with constraints
- ✅ Learn from user feedback to improve
- ✅ Scrape thecycle.online with full JavaScript rendering
- ✅ Explain its reasoning transparently
- ✅ Handle team filters and statistical constraints
- ✅ Provide confidence scores for answers

### Current Limitations
- ⚠️ **Data Source Issue**: Only gets default table view, not filtered/sorted data
- ⚠️ **UI Interaction**: Can't yet click filters and buttons on thecycle.online
- ⚠️ **CVR Access**: May not have access to CVR column in scraped data
- ⚠️ **Model Quality**: Using GPT-3.5-turbo (could upgrade to GPT-4o)

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- OpenAI API key
- Playwright browsers installed

### Quick Start

1. **Clone and setup:**
```bash
git clone <your-repo>
cd cycleAI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
playwright install chromium
```

3. **Configure API key:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

4. **Run the intelligent agent:**
```bash
streamlit run app.py
```

## 🎮 Usage Examples

### Intelligent Query Processing
```
Query: "which HOU player has the highest CVR with at least 80 games"

AI Reasoning:
- Semantic: Filter games_played >= 80
- Context: High confidence required, will include data reliability notes
- Learning: Column 'G' is reliable for games data
- Confidence: 78.5%

Answer: Based on Houston Astros players with 80+ games...
```

### Learning from Feedback
- Rate answers as 👍 good or 👎 bad
- AI learns which approaches work for different query types
- Performance improves over time

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_key
# Optional: Model selection
OPENAI_MODEL=gpt-4o  # Upgrade from gpt-3.5-turbo for better analysis
```

### Data Sources
- **Primary**: thecycle.online (JavaScript-rendered)
- **Fallback**: General web search via DuckDuckGo
- **Knowledge Base**: Local ChromaDB vector store

## 📈 Performance & Learning

### Current Metrics
- **Query Understanding**: ~75% confidence on complex queries
- **Data Extraction**: Successfully scrapes JavaScript sites
- **Learning Rate**: Adapts based on user feedback
- **Response Quality**: Improves with more interactions

### Learning Data
- Saved in `learned_knowledge.json`
- Query patterns and successful approaches
- Column mappings and data source preferences

## 🐛 Known Issues & Roadmap

### Immediate Issues
1. **Filter Interaction**: Can't click team filters or enable CVR column
2. **Data Completeness**: May not get all available stats
3. **UI Selectors**: Website selectors may change

### Planned Improvements
1. **Enhanced Scraping**: Better UI interaction and filter handling
2. **Model Upgrade**: GPT-4o for superior data analysis
3. **Multi-Source**: Combine data from multiple baseball sites
4. **Advanced Learning**: More sophisticated pattern recognition

## 🤝 Contributing

This is a portfolio project showcasing advanced AI concepts:
- **Semantic Understanding** vs rule-based systems
- **Learning from Feedback** for continuous improvement
- **Modern Web Scraping** with browser automation
- **Intelligent Agent Architecture** with reasoning

## 📄 License

Portfolio project - feel free to learn from and adapt the code.

---

**Built with**: LangChain, LangGraph, Playwright, Streamlit, ChromaDB, OpenAI
**Focus**: True AI intelligence beyond pattern matching

## Requirements

- OpenAI API key
- Internet connection for web search

## How it works

The agent uses tools that:
1. Enhances queries with "MLB" or "2025 MLB news" for relevance and recency.
2. Searches the web using DuckDuckGo (2 results for speed).
3. Fetches content from top results (5s timeout).
4. Splits text into 500-char chunks for finer retrieval.
5. Adds chunks to a Chroma vectorstore with timestamps (using local HuggingFace embeddings for speed).
6. Retrieves the most relevant recent documents (last 30 min).
7. Provides the context to the LLM for answering.
8. Cross-references with Wikipedia for verification.

Answers can be rated as good or bad, stored in `ratings.json` for future reference.

## Example

Question: Who won the World Series in 2023?

The agent will research, index web content, and provide the answer based on retrieved information.

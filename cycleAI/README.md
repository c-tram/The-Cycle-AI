# ⚾ Baseball AI Agent - Complete System

**An intelligent, multi-modal baseball analytics system with autonomous capabilities - from simple API queries to self-improving prompt generation.**

## 🎯 What This System Does

This is a **comprehensive baseball AI system** featuring:

- **🔥 Simple Mode**: Fast API-driven queries (players & teams data)
- **🧠 Advanced Mode**: LangChain-powered intelligent agent with tools  
- **🤖 Autonomous Mode**: Self-generating and testing prompt libraries
- **📊 Analytics Dashboard**: Performance monitoring and insights
- **Provides comprehensive statistics** for all 30 MLB teams and 1600+ players
- **Smart query processing** - automatically detects teams and stats in your questions
- **Real-time data** - always fetches the latest statistics
- **Clean, simple architecture** - no complex workflows or learning systems

## 🏗️ Current Architecture

### Simple & Focused Design

**Core Components:**
1. **🎯 Smart Query Parser** - Detects teams (HOU, NYY, BOS) and stats (hits, HR, RBI) in queries
2. **📊 API Client** - Fetches data from only 2 endpoints:
   - `https://thecycle.online/api/v2/players` (1600+ players)
   - `https://thecycle.online/api/v2/teams` (30 MLB teams)
3. **🧠 OpenAI Integration** - GPT-3.5-turbo for intelligent answers
4. **� Streamlit UI** - Clean web interface with detailed sidebar

### Data Flow
```
User Query → Smart Detection → API Fetch → Filter & Sort → AI Answer
```

## 🚀 Key Features

### Intelligence Features
- **Team Detection**: Recognizes all MLB team codes (HOU, NYY, LAD, etc.)
- **Stat Recognition**: Identifies hits, home runs, RBI, AVG, OPS, ERA, etc.
- **Smart Filtering**: Filters players/teams based on your query
- **Automatic Sorting**: Sorts by requested stat (most to least)
- **League Awareness**: AL vs NL team identification

### Technical Features
- **API-Only**: No web scraping, JavaScript rendering, or complex workflows
- **Real-Time Data**: Always fetches latest stats from APIs
- **Comprehensive Stats**: All available batting, pitching, and team statistics
- **Clean Output**: Focused, readable responses
- **Fast Performance**: Direct API calls with minimal processing

## 📊 Current Capabilities

### What It Can Do Well
- ✅ **Team-Specific Queries**: "HOU players by hits", "NYY batting stats"
- ✅ **Stat Rankings**: "most home runs", "highest batting average"
- ✅ **League Comparisons**: "AL teams by wins", "NL pitching leaders"
- ✅ **Player Lookups**: "Bo Bichette stats", "Shohei Ohtani performance"
- ✅ **Comprehensive Data**: All available stats from the APIs
- ✅ **Real-Time Updates**: Fresh data on every query

### Data Available
- **Players**: 1600+ MLB players with complete batting and pitching stats
- **Teams**: 30 MLB teams with records, batting, pitching, and fielding stats
- **Stats**: Hits, HR, RBI, AVG, OBP, SLG, OPS, ERA, WHIP, WAR, CVR, and more
- **Leagues**: Proper AL/NL identification for all teams

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- OpenAI API key

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
```

3. **Configure API key:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

4. **Run the agent:**
```bash
streamlit run app.py
```

## 🎮 Usage Examples

### Team-Specific Queries
```
Query: "List HOU players in order from most hits to least hits"

Response:
Jose Altuve (HOU, 2B) | 139 hits
Yordan Alvarez (HOU, LF) | 132 hits
Alex Bregman (HOU, 3B) | 128 hits
...
```

### Stat Rankings
```
Query: "Who leads the league in home runs?"

Response:
Aaron Judge (NYY, RF) leads with 42 home runs, followed by Shohei Ohtani (LAD, P) with 38.
```

### League Comparisons
```
Query: "AL teams by wins"

Response:
Shows all AL teams sorted by wins with their records and stats
```

### Player Lookups
```
Query: "Bo Bichette stats"

Response:
Bo Bichette (TOR, SS) | G:133, AB:528, R:97, H:171, 2B:40, 3B:2, HR:20, RBI:79, SB:20, BB:35, SO:102, AVG:.324, OBP:.356, SLG:.511, OPS:.867
```

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_key_here
```

### API Endpoints
- **Players API**: `https://thecycle.online/api/v2/players`
- **Teams API**: `https://thecycle.online/api/v2/teams`

## 📈 Smart Query Processing

### How It Detects Your Intent

The agent automatically detects:

**Teams**: HOU, NYY, BOS, LAD, TOR, ATL, etc.
**Stats**: hits, home runs, RBI, AVG, OPS, ERA, WHIP, etc.
**Leagues**: AL, NL team identification
**Sorting**: most/least, highest/lowest, top/bottom

### Query Examples
- ✅ "HOU players by hits" → Filters HOU players, sorts by hits
- ✅ "most home runs" → Finds player with highest HR
- ✅ "AL teams by wins" → Shows AL teams sorted by wins
- ✅ "Bo Bichette doubles" → Shows Bo Bichette's doubles stat

## 🎯 Performance & Reliability

### Current Metrics
- **API Reliability**: 100% success rate with thecycle.online APIs
- **Query Accuracy**: Precise filtering and sorting based on detected intent
- **Data Freshness**: Real-time data on every request
- **Response Speed**: Fast API calls with minimal processing

### Data Quality
- **Complete Coverage**: All 30 MLB teams and 1600+ players
- **Rich Statistics**: Comprehensive batting, pitching, and team stats
- **League Accuracy**: Proper AL/NL identification for all teams
- **Stat Reliability**: Direct from official API data

## 🆚 Why This Approach Works

### Before (Complex System)
- ❌ Multiple intelligence components
- ❌ JavaScript web scraping
- ❌ Complex LangGraph workflows
- ❌ Learning systems and feedback loops
- ❌ Browser automation with Playwright
- ❌ Multiple data sources and fallbacks

### After (API-Only)
- ✅ **Simple, focused architecture**
- ✅ **Direct API integration**
- ✅ **Reliable data access**
- ✅ **Fast performance**
- ✅ **Easy maintenance**
- ✅ **Clear, predictable behavior**

## 🚀 Advanced Implementation Options

Your current simple API-only architecture provides an excellent foundation. Here are **optional** advanced features you can add incrementally:

### 1. Enhanced Data Processing (`app_enhanced.py`)
- **Caching System**: 30-minute data caching to reduce API calls
- **Advanced Query Analysis**: Better intent detection and parameter extraction
- **Structured Data Classes**: Type-safe data handling
- **Improved Filtering**: More sophisticated player/team filtering

### 2. LangChain Integration (`langchain_integration.py`)
- **Modular Tools**: Convert API calls into reusable LangChain tools
- **Intelligent Agent**: Multi-step analysis workflows
- **Memory Management**: Conversation context across queries
- **Custom Prompts**: Baseball-specific prompt engineering

### 3. Advanced Features Available
- **Vector Search**: Semantic search across player statistics
- **Multi-step Workflows**: Complex analysis chains
- **Performance Caching**: Redis-backed caching for high traffic
- **Advanced Analytics**: Comparative analysis and insights

### 4. Easy Integration
```bash
# Install advanced dependencies (optional)
pip install -r requirements_advanced.txt

# Run enhanced version
streamlit run app_enhanced.py

# Or use LangChain agent
python langchain_integration.py
```

### 5. Why These Enhancements Work
- **Incremental**: Add features without breaking current functionality
- **Optional**: Current simple version remains the core
- **Scalable**: Architecture supports growth
- **Maintainable**: Clean separation of concerns

## 🆚 Why This Approach Works

### Before (Complex System)
- ❌ Multiple intelligence components
- ❌ JavaScript web scraping
- ❌ Complex LangGraph workflows
- ❌ Learning systems and feedback loops
- ❌ Browser automation with Playwright
- ❌ Multiple data sources and fallbacks

### After (API-Only)
- ✅ **Simple, focused architecture**
- ✅ **Direct API integration**
- ✅ **Reliable data access**
- ✅ **Fast performance**
- ✅ **Easy maintenance**
- ✅ **Clear, predictable behavior**

## 🤝 Contributing

This is a portfolio project demonstrating:
- **API Integration**: Clean, reliable data access
- **Smart Query Processing**: Intent detection and filtering
- **Data Analysis**: Statistical ranking and comparison
- **User Experience**: Intuitive query handling
- **Performance**: Fast, reliable responses

## 📄 License

Portfolio project - feel free to learn from and adapt the code.

---

**Built with**: Streamlit, OpenAI GPT-3.5-turbo, Requests
**Focus**: Multi-modal baseball AI with autonomous learning capabilities

## 🚀 Quick Start

### Option 1: Simple Mode (Fast)
```bash
./run_agent.sh
# Choose option 1 for simple mode
```

### Option 2: Advanced Mode (Smart)
```bash
./run_advanced.sh
# Full LangChain integration with tools
```

### Option 3: Autonomous Mode (Self-Improving)
```bash
./run_autonomous.sh
# CLI autonomous prompt generation
```

### Option 4: Autonomous Dashboard (Interactive)
```bash
./run_autonomous_ui.sh
# Full Streamlit interface on port 8502
```

## 🔧 System Modes

### ⚡ Simple Mode
- **Fast API-only** queries
- **Direct responses** for basic questions
- **Minimal processing** overhead
- **Perfect for**: Quick stats, team queries, player lookups

### 🧠 Advanced Mode  
- **LangChain-powered** intelligent reasoning
- **Tool selection** based on query complexity
- **Memory and context** awareness
- **Perfect for**: Complex analysis, comparisons, team building

### 🤖 Autonomous Mode
- **Self-generating** prompt libraries
- **Automatic testing** and quality assessment
- **Performance analytics** and insights
- **Continuous improvement** through iteration
- **Perfect for**: System optimization, prompt discovery, scaling

## 🛠️ Available Tools (Advanced/Autonomous)

1. **🔍 FetchPlayersTool** - Comprehensive player data retrieval
2. **🏟️ FetchTeamsTool** - Complete team statistics and records  
3. **📊 AnalyzeTeamPerformanceTool** - Deep team performance analysis
4. **🔤 FilterPlayersByLastNameTool** - Name-based player filtering
5. **🏆 BuildBestTeamTool** - Optimal team construction with explanations

## 🤖 Autonomous Features

### Prompt Generation Engine
- **10 Template Categories**: Player stats, team analysis, comparisons, etc.
- **Variable Substitution**: Automatic parameter variation
- **Complexity Levels**: Simple, medium, complex query types
- **Quality Assessment**: Automated response evaluation

### Performance Analytics
- **Success Rate Tracking**: Monitor prompt effectiveness
- **Category Analysis**: Performance by query type
- **Tool Usage Metrics**: Understand system utilization
- **Quality Scoring**: Response assessment algorithms

### Continuous Learning
- **Batch Processing**: Generate and test multiple prompts
- **Iterative Improvement**: Learn from successful patterns
- **Library Management**: Export/import prompt collections
- **Recommendation Engine**: Suggest optimization strategies

## Requirements

- Python 3.9+
- OpenAI API key
- Internet connection for API access
- Advanced features: `pip install -r requirements_advanced.txt`

## How It Works

### Simple Mode Flow
1. **Query Analysis** → **API Fetching** → **Smart Filtering** → **AI Response**

### Advanced Mode Flow  
1. **Query Analysis** → **Tool Selection** → **Multi-step Reasoning** → **Contextual Response**

### Autonomous Mode Flow
1. **Template Selection** → **Variable Substitution** → **Batch Generation** → **Automated Testing** → **Quality Assessment** → **Library Update**

## Example Queries by Mode

### Simple Mode Examples
- "HOU players by hits" - Houston players sorted by hits
- "AL teams by ERA" - AL teams sorted by team ERA
- "Shohei Ohtani stats" - Complete player statistics

### Advanced Mode Examples  
- "Build the best baseball team with every position filled"
- "Compare Yankees and Dodgers pitching staffs this season"
- "Show me players whose last name starts with T with over 20 home runs"

### Autonomous Mode Examples
*Auto-generated prompts include:*
- "Who has the most stolen bases in their career?"
- "How is the Braves performing in defense this season?"
- "Show me all players whose last name starts with M who have more than 50 RBIs"
- "top batting averages" - Highest batting averages in MLB

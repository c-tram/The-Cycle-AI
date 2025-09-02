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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key from .env
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize embeddings and vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Custom tool for web research with RAG
@tool
def web_research_tool(query: str) -> str:
    """
    Researches the web for the given query, fetches content from top results,
    indexes it using RAG, and retrieves relevant information.
    """
    # Enhance query for recency and relevance
    enhanced_query = query
    if "this year" in query.lower() or "current" in query.lower() or "2025" in query.lower():
        enhanced_query += " 2025 MLB news"
    else:
        enhanced_query += " MLB"
    
    # Search the web (reduced to 2 results for speed)
    with DDGS() as ddgs:
        results = ddgs.text(enhanced_query, max_results=2)    # Fetch content from top results
    # Fetch content from top results (shorter timeout)
    docs = []
    for r in results:
        url = r['href']
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            docs.append(text)
        except Exception as e:
            continue
    
    # Split text into chunks
    # Split text into smaller chunks for finer retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = []
    metadatas = []
    current_time = time.time()
    for doc in docs:
        splits = text_splitter.split_text(doc)
        all_splits.extend(splits)
        metadatas.extend([{"timestamp": current_time} for _ in splits])
    
    # Add to vectorstore with metadata
    if all_splits:
        vectorstore.add_texts(all_splits, metadatas=metadatas)
    
    # Retrieve relevant documents, filtering to recent (last 1 hour)
    # Retrieve relevant documents (last 30 min for fresher data)
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"timestamp": {"$gte": current_time - 1800}}  # Last 30 min
        }
    )
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    return context

# Initialize Wikipedia tool
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# List of tools
tools = [web_research_tool, wikipedia_tool]

# Define the prompt template
template = """
You are a helpful AI assistant specialized in baseball. You can answer any question about baseball by researching the web and verifying information using available tools. Always cross-reference information from multiple sources when possible to ensure accuracy. Note the recency of the information and any potential uncertainties.

Use the following tools if needed:

{tools}

Use the following format:

Question: the input question
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}

{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    question = input("Ask a baseball question: ")
    result = agent_executor.invoke({"input": question})
    print(result["output"])

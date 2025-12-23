## External tools that the llm is going to access

from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv,description="Query arxiv papers")
#print(arxiv.name)  # To check if the tool is successfully created

""" result = arxiv.invoke("What is the latest research on AI computing?")  ### check wheather or not the arxiv is working
print(result) """

api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
wiki.name

### IMPORTING THE API KEYS FROM THE ENV FILE
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")  
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

### Tavily Search Tool (Just like Google but for LLMs)
from langchain_community.tools.tavily_search import TavilySearchResults
tavily = TavilySearchResults()
""" result = tavily.invoke("Provide me the recent AI news?")      ### check wheather or not the tavily is working
print(result) """

## combine all these tools in the list so we can integrate with llm
tools=[arxiv, wiki, tavily]

## Initialize the LLM Model
from langchain_groq import ChatGroq
llm=ChatGroq(model="qwen/qwen3-32b")
""" result = llm.invoke("What is AI")             ### check wheather or not the llm is working
print(result) """

### To Bind or combine our llm with these tools
llm_with_tools=llm.bind_tools(tools=tools)

## State Schema
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage ## Human message or AI message
from typing import Annotated  ## labelling
from langgraph.graph.message import add_messages  ## Reducers in Langgraph (for appending AI and Human Messages)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  
    
### Entire Chatbot With LangGraph
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

### Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

## Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

### To view the graph
""" png_bytes = graph.get_graph().draw_mermaid_png()

with open("BotFlow.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as BotFlow.png") """

messages=graph.invoke({"messages":"Hi, my name is Wali, How are you? and then please tell me the recent research paper on quantum computing?"})
for m in messages['messages']:
    m.pretty_print()
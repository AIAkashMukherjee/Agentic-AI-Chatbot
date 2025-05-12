import os

# step 1 api keys

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


# step 2 setup llm and tools 
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

open_ai=ChatOpenAI(model='gpt-4o-mini')
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

search_tool=TavilySearchResults(max_result=2)

# step 3 setup ai agent 
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt='Act as a AI chatbot who is smart'


# create a function for this
def create_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id)

    tools=[TavilySearchResults(max_result=2)]if allow_search else [] 

    agent=create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=system_prompt
    )

    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]




# agent=create_react_agent(
#     model=groq_llm,
#     tools=[search_tool],
#     state_modifier=system_prompt
# )
# query='Tell me about AI'
# state={"message":query}

# response=agent.invoke(state)
# messages=response.get('messages')

# ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]


# print(ai_messages[-1])
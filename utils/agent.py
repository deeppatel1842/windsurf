from typing import Dict, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
# from langgraph.prebuilt import ToolExecutor
# from langgraph.prebuilt.tool_executor import ToolExecutor

from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_step: str

def create_agent():
    # Initialize the language model
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that helps users find information."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])

    # Create the chain
    chain = prompt | model

    return chain

def should_continue(state: AgentState) -> bool:
    """
    Determine if the conversation should continue.
    """
    return state["current_step"] != "end"

def get_next_step(state: AgentState) -> str:
    """
    Determine the next step in the conversation.
    """
    # Add your logic here to determine the next step
    return "end"

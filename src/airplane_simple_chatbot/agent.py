from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from airplane_simple_chatbot.config import model, config
from airplane_simple_retriever.schemas import State as RetrieverState

def agent_function(messages: list[BaseMessage] ):
    prompt = ChatPromptTemplate.from_messages(config.main.to_list()+ messages)
    chain = prompt | model.with_config(run_name="stream")  
    return chain

def chatbot(state: RetrieverState):
    documents = state.retriver_results if len(state.retriver_results) > 0 else "no documents found, we don't have the answer to the question, do not provide any additional information."
    chain = agent_function(state.messages)
    return chain.invoke({"documents": documents})
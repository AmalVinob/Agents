from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere

# Creating the first analysis agent to check the prompt structure
# This print part helps you to trace the graph decisions

def analyze_question(state):
    llm = ChatCohere()
    prompt = PromptTemplate.from_template("""
    You are an agent that needs to define if a question is a technical code one or a general one.

    Question : {input}

    Analyse the question. Only answer with "code" if the question is about technical development. If not just answer "general".

    Your answer (code/general) :
    """)
    chain = prompt | llm
    response = chain.invoke({"input": state['input']})
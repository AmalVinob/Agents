import operator
from typing import Annotated
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles



class State(TypedDict):
    report_url : str
    financial_metric : dict
    sentimental_analysis : str
    intermediate_sentiment_report : str
    final_decision_report : str
    aggregate : Annotated[list, operator.add]


def finanacial_metric_extraction(state: State):
    report_url = State["report_url"]

    extrated_metrics = {"revenue" : 10, "ESP" :2} 
    return {"finanacial_metric" : extrated_metrics, "aggregate" : [extrated_metrics]}

def sentimental_analyser(state: State):
    report_url = State["report_url"]

    sentimental_analysis = "line 3 is positive line 45 is very negative"
    return {"sentimental_analysis": sentimental_analysis, "aggregate" : [sentimental_analysis]}

def sentimental_repo_gen(state: State):
    sentimental_analysis= state["sentimental_analysis"]

    intermediate_sentiment_report = "buy because following pos, sell because follow neg"
    return {"intermediate_sentimental": intermediate_sentiment_report, "aggregate" : [intermediate_sentiment_report]}


def final_report_generator(state: State):
    financial_metric = State['financial_metric']
    sentimental_analysis = State["intermediate_sentiment_report"]

    final_decision_report = "crazy report is here "
    return {"final_decision_report" : final_decision_report,  "aggregate" : [final_decision_report]}


graph_builder = StateGraph(State)

graph_builder.add_node("mtrc_ext", finanacial_metric_extraction)
graph_builder.add_node("snt_ext", sentimental_analyser)
graph_builder.add_node("snt_repo", sentimental_repo_gen)
graph_builder.add_node("fnl_rprt", final_report_generator)

graph_builder.add_edge(START, "mtrc_ext")
graph_builder.add_edge(START, "snt_ext")
# graph_builder.add_edge("mtrc_ext", "fnl_rprt")
graph_builder.add_edge("snt_ext", "snt_repo")
graph_builder.add_edge(["mtrc_ext", "snt_repo"], "fnl_rprt")
graph_builder.add_edge("fnl_rprt", END)

graph = graph_builder.compile()


print(graph.get_graph().draw_mermaid())
display(
    Image(
        graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)

result = graph.invoke({"report_url" : "test.com"})
print(result)
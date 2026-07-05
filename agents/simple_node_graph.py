from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    message: str


def process_message(state: AgentState) -> AgentState:
    """Prepends 'Hello' to the message in AgentState."""
    state["message"] = f"Hello {state['message']}"
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process_message)
graph.add_edge(START, "process")
graph.add_edge("process", END)

if __name__ == "__main__":
    app = graph.compile(name="simple_greeting_graph")
    response = app.invoke({'message':'Bimal'})
    print(response)

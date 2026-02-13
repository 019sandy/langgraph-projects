from dotenv import load_dotenv
load_dotenv()
from typing import List, Sequence, TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from chains import generation_chain , reflection_chain
import os

class GraphState(TypedDict):
    messages: Sequence[BaseMessage]


REFLECT = "reflect"
GENERATE = "generate"

# 1. Define Generation Node -> It recieve the State history
def generation_node(state: Sequence[BaseMessage]):
    return generation_chain.invoke({"messages": state})

# 2. Define Reflection node -> it receievs the history in messages a salso from above
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflection_chain.invoke({"messages": messages})
    # store it asd we will trick AI by making it feel like human message , whih it has to refine
    return [HumanMessage(content=res.content)]
    
# 3. Initialize graph and add nodes
builder = StateGraph(state_schema=GraphState)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

# 4. write sould continue node 
def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END 
    else:
        return REFLECT
    
# 5. Add conditional Edge
builder.add_conditional_edges(GENERATE, should_continue, {END:END, REFLECT:REFLECT})

# 6. Add Edge
builder.add_edge(GENERATE, REFLECT)


# 7. Compile and Visualize
graph = builder.compile()
print(graph.get_graph().draw_mermaid_png)
open("graph.png", "wb").write(graph.get_graph().draw_mermaid_png())



if __name__ == "__main__":
    pass


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
def generation_node(state: GraphState):
    res = generation_chain.invoke({"messages": state["messages"]})
    # because state is a Dict we need to retrun it this way
    return {"messages": state["messages"] + [res]}

# 2. Define Reflection node -> it receievs the history in messages a salso from above
def reflection_node(state: GraphState):
    res = reflection_chain.invoke({"messages": state["messages"]})
    # store it asd we will trick AI by making it feel like human message , whih it has to refine
    return {"messages": state["messages"] + [HumanMessage(content=res.content)]}
    
# 3. Initialize graph and add nodes
builder = StateGraph(state_schema=GraphState)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

# 4. write sould continue node 
def should_continue(state: GraphState):
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
    # 8. lets run the functionality
    print("Hello Langgraph User")

    inputs = HumanMessage(content="""Make this tweet better:"
                          At Seva Teerth earlier today, signed files that are connected with the empowerment of the poor, downtrodden, our hardworking farmers, Yuva Shakti and Nari Shakti. The key decisions include:
        Approval for the launch of PM RAHAT scheme, as per which victims of accidents will get cashless treatment of up to Rs. 1.5 lakhs. This will ensure that immediate medical help is provided to anyone affected. 
        Doubling of the target of the Lakhpati Didi scheme to 6 crore. This will boost women empowerment and self-reliance. 
        The Agriculture Infrastructure Fund target has been doubled from Rs. 1 lakh crore to Rs. 2 lakh crore. This will be immensely helpful to our farmers and help boost their incomes.
        Approval for a Startup India Fund of Funds 2.0 with a corpus of Rs. 10,000 crore. This will encourage startups in early stages and deep-tech research.
        These decisions reaffirm the spirit of Nagrikdevo Bhava and add momentum to our vision of building a Viksit Bharat.""")
    
    response = graph.invoke(
        {
            "messages": [inputs]
        }
    )



from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a reflection agent that analyzes the past tweetes."
            "Generate a Critique for the user depending uopn length , virality , engagement and style "
            "Always provide the detailed feedback and recommendations"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
 
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a generation agent who is a twitter techine influencer that generates new tweets."
            "Generate the best possible tweet for user's request"
            "If the user provides the critique generate the tweet based on the critique."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(temperature=0, model_name="gpt-4")

reflection_chain = reflection_prompt | llm   # bind_llm(llm) # or reflection_prompt | llm
generation_chain = generation_prompt | llm    # bin_llm(llm) # generation_prompt | llm


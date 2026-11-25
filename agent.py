import os
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, OpenAIServerModel, DuckDuckGoSearchTool

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found. Check your .env file.")

model = OpenAIServerModel(
    model_id="openrouter/free",
    api_base="https://openrouter.ai/api/v1",
    api_key=api_key,
    flatten_messages_as_text=True,
)

agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    max_steps=5,
)

if __name__ == "__main__":
    question = "What are the main differences between LightGBM and XGBoost? Give 3 concise bullet points with sources."
    result = agent.run(question)
    print("\n=== FINAL ANSWER ===\n")
    print(result)
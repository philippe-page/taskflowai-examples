from taskflowai import Task, Agent, WebTools, OpenrouterModels

# Define the web_researcher
web_researcher = Agent(
    role="web researcher",
    goal="to search the web and provide accurate information",
    attributes="thorough, detail-oriented, and able to distill complex information.",
    llm=OpenrouterModels.sonnet_3_5,
    tools=[WebTools.exa_search]
)

# Define the simple research and respond task
def research_and_respond_task(web_researcher, input):
    return Task.create(
        agent=web_researcher,
        instruction=f"Use your web search tool to answer the user query: '{input}' in a conversational way. You MUST call the tool multiple times to create unique searches, for 5-10 results, sentences and snippets each. Cite your sources in your response. Your response should be comprehensive, detailed, and written in rich markdown format."
    )

def main():
    user_query = input("Hey, ask away!\n")
    response = research_and_respond_task(web_researcher, user_query)
    print(f"Final response: {response}")

if __name__ == "__main__":
    main()
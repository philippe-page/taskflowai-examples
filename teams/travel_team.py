from taskflowai import Agent, Task, WebTools, WikipediaTools, AmadeusTools, OpenaiModels, OpenrouterModels, set_verbosity

set_verbosity(True)

web_research_agent = Agent(
    role="web research agent",
    goal="search the web thoroughly for travel information",
    attributes="hardworking, diligent, thorough, comphrehensive.",
    llm=OpenrouterModels.sonnet_3_5,
    tools=[WebTools.serper_search, WikipediaTools.search_articles, WikipediaTools.search_images]
)

travel_agent = Agent(
    role="travel agent",
    goal="assist the traveller with their request",
    attributes="frindly, hardworking, and comprehensive and extensive in reporting back to users",
    llm=OpenrouterModels.sonnet_3_5,
    tools=[AmadeusTools.search_flights, WebTools.get_weather_data]
)

def research_destination(destination, interests):
    destination_report = Task.create(
        agent=web_research_agent,
        context=f"User Destination: {destination}\nUser Interests: {interests}",
        instruction=f"Use your tools to search relevant information about the given destination: {destination}. Use your serper web search tool to research information about the destination to write a comprehensive report. Use wikipedia tools to search the destination's wikipedia page, as well as images of the destination. In your final answer you should write a comprehensive report about the destination with images embedded in markdown."
    )
    return destination_report

def research_events(destination, dates, interests):
    events_report = Task.create(
        agent=web_research_agent,
        context=f"User's intended destination: {destination}\n\nUser's intended dates of travel: {dates}\nUser Interests: {interests}",
        instruction="Use your tools to research events in the given location for the given date span. Ensure your report is a comprehensive report on events in the area for that time period."
    )
    return events_report

def search_flights(current_location, destination, dates):
    flight_report = Task.create(
        agent=travel_agent,
        context=f"Current Location: {current_location}\n\nDestination: {destination}\nDate Range: {dates}",
        instruction=f"Search for a lot of flights in the given date range to collect a bunch of options and return a report on the best options in your opinion, based on convenience and lowest price."
    )
    return flight_report

def write_travel_report(destination_report, events_report, flight_report):
    travel_report = Task.create(
        agent=travel_agent,
        context=f"Destination Report: {destination_report}\n--------\n\nEvents Report: {events_report}\n--------\n\nFlight Report: {flight_report}",
        instruction=f"Write a comprehensive travel plan and report given the information above. Ensure your report conveys all the detail in the given information, from flight options, to weather, to events, and image urls, etc. Preserve detail and write your report in extensive length."
    )
    return travel_report

def main():
    current_location = input("Where are you traveling from?\n")
    destination = input("Where are you travelling to?\n")
    dates = input("What are the dates for your trip?\n")
    interests= input("Do you have any particular interests?\n")

    destination_report = research_destination(web_research_agent, destination, interests)
    print(destination_report)

    events_report = research_events(web_research_agent, destination, dates, interests)
    print(events_report)

    flight_report = search_flights(travel_agent, current_location, destination, dates)
    print(flight_report)

    final_report = write_travel_report(travel_agent, destination_report, events_report, flight_report)
    print(final_report)

if __name__ == "__main__":
    main()
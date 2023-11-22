import streamlit as st
import pandas as pd
import heapq
import matplotlib.pyplot as plt

# Assuming the CSV is in the same directory as the Streamlit app for simplicity
# Otherwise, you need to upload it or point to the correct path
@st.cache
def load_data(csv_file):
    return pd.read_csv(csv_file, delimiter='\t')

# Function to build the graph from DataFrame
def build_graph(df):
    graph = {}
    for _, row in df.iterrows():
        trip_id = row['trip_id']
        trajet = row['trajet']
        time = row['duree']
        cities = trajet.split(" - ")
        if len(cities) == 2:
            city1, city2 = cities
            if city1 not in graph:
                graph[city1] = {}
            if city2 not in graph:
                graph[city2] = {}
            graph[city1][city2] = time
            graph[city2][city1] = time
    return graph

# Dijkstra's algorithm implementation
def dijkstra(graph, start_city, end_city):
    distances = {city: float('inf') for city in graph}
    distances[start_city] = 0
    previous = {city: None for city in graph}
    unvisited_cities = [(0, start_city)]

    while unvisited_cities:
        current_distance, current_city = heapq.heappop(unvisited_cities)
        if current_distance > distances[current_city]:
            continue

        for neighbor, time in graph[current_city].items():
            time_to_neighbor = distances[current_city] + time
            if time_to_neighbor < distances[neighbor]:
                distances[neighbor] = time_to_neighbor
                previous[neighbor] = current_city
                heapq.heappush(unvisited_cities, (time_to_neighbor, neighbor))

    path = []
    current = end_city
    while current is not None:
        path.insert(0, current)
        current = previous[current]

    return distances[end_city], path

# Function to plot the shortest path
def plot_shortest_path(stations, durations):
    fig, ax = plt.subplots(figsize=(10, 2))
    for i in range(len(stations) - 1):
        from_station = stations[i]
        to_station = stations[i + 1]
        duration = durations[i]
        ax.plot([i, i + 1], [0, 0], 'r-')
        ax.annotate(f"{duration} min", ((i + i + 1) / 2, 0), ha='center', va='bottom')
        ax.plot(i, 0, 'ro', markersize=8, label=f'{from_station}')
    ax.plot(len(stations) - 1, 0, 'ro', markersize=8, label=f'{stations[-1]}')
    ax.set_xticks(range(len(stations)))
    ax.set_xticklabels(stations, rotation=45, fontsize=8)
    ax.set_title("Shortest Path with Durations")
    ax.set_xlabel("Stations")
    ax.set_ylabel("Duration (minutes)")
    plt.tight_layout()
    return fig

# Streamlit App
st.title("Path finder in France")

# Load data and build the graph
df = load_data('timetables.csv')  
graph = build_graph(df)

# Select boxes for start and end cities
start_city = st.selectbox("Select the starting city:", options=list(graph.keys()))
end_city = st.selectbox("Select the destination city:", options=list(graph.keys()))

# Find path button
if st.button("Find Shortest Path"):
    shortest_time, shortest_path = dijkstra(graph, start_city, end_city)
    if shortest_path:
        st.write(f"Shortest time from {start_city} to {end_city}: {shortest_time} minutes")
        durations = [graph[shortest_path[i]][shortest_path[i + 1]] for i in range(len(shortest_path) - 1)]
        fig = plot_shortest_path(shortest_path, durations)
        st.pyplot(fig)
    else:
        st.error("No valid path found.")

# TSP-ACO
An Ant Colony Optimization implementation for solving the TSP problem
This is a solution to the Kaggle's challenge named "Traveling Santa Problem" that can be found here: https://www.kaggle.com/c/traveling-santa-problem.

We have provided the kaggle original CSV file (santa_cities_full.csv) and a smaller file with 50 cities only (santa_cities_50.csv).

The solution is composed of 3 Python files:
- TSP5explore.py
- TSP5_hunter.py
- TSP5Libs.py
- print_results.py

The first file implement the exploring ant. It gets the CSV source file as input (must be on the same directory), and serialize the graph to a file on the same directory. Exploring ants are used to initialize the graph by finding the most useful edges.

The second file implements the hunting ant. It requires a graph serialized by TSP5explorer.py to a file on the same directory, and produce as output the serialization to disk of a graph containing the 2 solutions. Hunting ants are used to hunt for solutions.

The explorer takes longer to run, cause it uses a library to query nodes by distance, and cause the hunter employs a clustering mechanism that after some iterations improve the computing efficiency. For this reason a graph serialized by the exploring ant is provided. In order to print the 2 cheminies found by the explorer the print_results.py file should be executed.

TSP5Libs.py contains some common libraries.

The solution runs on Python 3.6.

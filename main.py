import math
import itertools


def calc_dists(n1, n2):
    return math.sqrt(
        (n1.coords[0] - n2.coords[0]) ** 2 + (n1.coords[1] - n2.coords[1]) ** 2
    )


class Node:
    def __init__(self, coords, name):
        self.name = name
        self.coords = coords
        self.dists = {}
        self.pheremones = {}

    def __str__(self):
        return f"{self.coords}"

    def show_dists(self):
        for node in self.dists.keys():
            print(f"Dist from {self} to {node} is {self.dists[node]}")


class Ant:
    def __init__(self):
        self.nodes_visited = []


def populate_dists():
    for n1, n2 in itertools.combinations(nodes, 2):
        dist = calc_dists(n1, n2)
        n1.dists[n2] = dist
        n2.dists[n1] = dist


nodes = [
    Node((0.0, 0.0), "A"),
    Node((1.0, 0.0), "B"),
    Node((0.0, 1.0), "C"),
    Node((1.0, 1.0), "D"),
]


populate_dists()

for node in nodes:
    node.show_dists()

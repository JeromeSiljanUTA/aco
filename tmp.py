# Do this in cuda bro

# Calculate desires
for node in range(NUM_NODES):
    for desire_node in range(NUM_NODES):
        current_node = ant_matrix[ant][CURRENT_NODE_COL]

        desire_node_visited = False
        # Has desire node not been marked visited
        for i in range(NUM_NODES):
            if desire_node == prev_visited_matrix[ant][i]:
                desire_node_visited = True
                break

        if desire_node != current_node and not desire_node_visited:
            desire = ((pheromones[current_node][desire_node]) ** ALPHA) * (
                (1 / distances[current_node][desire_node]) ** BETA
            )

            desires_matrix[ant][desire_node] = desire

            if DEBUG:
                print(f"Comparing {current_node} and {desire_node}: {desire}")

        else:
            if DEBUG:
                print(f"Not comparing {current_node} and {desire_node}")

            desires_matrix[ant][desire_node] = 0

    if node != NUM_NODES - 1:
        probability_matrix[ant] = desires_matrix[ant] / np.sum(desires_matrix[ant])
        # Choosing next node
        target_node = np.random.choice(outcomes, p=probability_matrix[ant])
        current_node_visited = False
        target_node_visited = False

        # Has current node not been marked visited
        for i in range(NUM_NODES):
            if ant_matrix[ant][CURRENT_NODE_COL] != prev_visited_matrix[ant][i]:
                current_node_visited = True

        if not current_node_visited:
            for i in range(NUM_NODES):
                if prev_visited_matrix[ant][i] == -1:
                    prev_visited_matrix[ant][i] = ant_matrix[ant][CURRENT_NODE_COL]

        # Has target_node been marked visited?
        for i in range(NUM_NODES):
            if target_node == prev_visited_matrix[ant][i]:
                target_node_visited = True

        target_node_set = False
        if not target_node_visited:
            for i in range(NUM_NODES):
                if prev_visited_matrix[ant][i] == -1 and not target_node_set:
                    prev_visited_matrix[ant][i] = target_node
                    ant_matrix[ant][CURRENT_NODE_COL] = target_node
                    target_node_set = True

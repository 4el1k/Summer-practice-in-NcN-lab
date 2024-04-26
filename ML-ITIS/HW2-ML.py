import pygame
import numpy as np
import sys

pygame.init()

# Window settings
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
DISPLAY_WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("DBSCAN Simulation")

# Define colors
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_ORANGE = (255, 165, 0)
COLOR_PURPLE = (128, 0, 128)

# Define point radius
POINT_RADIUS = 5

all_points = []

def plot_point(surface, location, color=COLOR_BLACK):
    pygame.draw.circle(surface, color, location, POINT_RADIUS)

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def apply_dbscan(data_points, epsilon, minimum_points):
    cluster_data = []
    outliers = []
    processed = set()
    classifications = {}  # 0 - outlier, 1 - border, 2 - core

    def query_region(p):
        close_points = []
        for other in data_points:
            if calculate_distance(p, other) < epsilon:
                close_points.append(other)
        return close_points

    def form_cluster(initial, nearby, cluster_group):
        classifications[initial] = 2  # Mark as core
        cluster_group.append(initial)
        while nearby:
            current = nearby.pop()
            if current not in processed:
                processed.add(current)
                nearby_neighbors = query_region(current)
                if len(nearby_neighbors) >= minimum_points:
                    nearby.extend(nearby_neighbors)
                    classifications[current] = 2  # Mark as core
                else:
                    classifications[current] = 1  # Mark as border
            if current not in sum(cluster_data, []):  # Ensure not already in any cluster
                cluster_group.append(current)

    for single_point in data_points:
        if single_point not in processed:
            processed.add(single_point)
            local_neighbors = query_region(single_point)
            if len(local_neighbors) < minimum_points:
                outliers.append(single_point)
                classifications[single_point] = 0  # Mark as noise
            else:
                new_cluster = []
                form_cluster(single_point, local_neighbors, new_cluster)
                cluster_data.append(new_cluster)

    return cluster_data, outliers, classifications

# Main loop
active = True
try:
    while active:
        DISPLAY_WINDOW.fill(COLOR_WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                active = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse click
                    mouse_position = pygame.mouse.get_pos()
                    all_points.append(mouse_position)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and all_points:
                    epsilon_value = 50
                    minimum_value = 3
                    clusters_result, noise_result, points_classifications = apply_dbscan(all_points, epsilon_value, minimum_value)

                    # Visualize points based on their classification
                    DISPLAY_WINDOW.fill(COLOR_WHITE)
                    points_color_map = {0: COLOR_YELLOW, 1: COLOR_RED, 2: COLOR_GREEN}
                    for each_point in all_points:
                        plot_point(DISPLAY_WINDOW, each_point, points_color_map[points_classifications[each_point]])

                    pygame.display.update()

                    # Wait to visualize
                    pygame.time.wait(2000) # Visualize clusters
                    cluster_color_palette = [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, COLOR_CYAN, COLOR_MAGENTA, COLOR_ORANGE, COLOR_PURPLE]
                    DISPLAY_WINDOW.fill(COLOR_WHITE)
                    for index, group in enumerate(clusters_result):
                        color_selection = cluster_color_palette[index % len(cluster_color_palette)]
                        for member_point in group:
                            plot_point(DISPLAY_WINDOW, member_point, color_selection)
                    pygame.display.update()

        for pt in all_points:
            plot_point(DISPLAY_WINDOW, pt)

        pygame.display.update()

except KeyboardInterrupt:
    active = False
finally:
    pygame.quit()
    sys.exit()
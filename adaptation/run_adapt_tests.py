import sys
import os
import numpy as np
import csv
from .adapt_tests_file_utils import save_results_to_csv, save_average_performance_to_csv

sys.path.append(os.path.abspath("."))

from MBOA import MBOA

# Parameters
map_count = 1
niches = 5  # k
SHOW_VISUAL = False
runs = 20  # Number of runs to average over

# Failure scenarios
scenarios = [
    [[]],
    [[1], [2], [3], [4], [5], [6]],
    [[1, 4], [2, 5], [3, 6]],
    [[1, 3], [2, 4], [3, 5], [4, 6], [5, 1], [6, 2]],
    [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]]
]

niches = [5000, 10000, 20000, 40000]

def get_centroids_path(niches):
	return os.path.join(os.path.dirname(os.path.dirname(__file__)), "centroids", f"centroids_{niches}_6.dat")

def get_map_path(niches, map_num):
	'''
	Returns the path to the map file for the specified number of niches
	and map number
	:param niches: Number of niches in the map
	:param map_num: Map number. This is just to identify different maps for the same number of niches
	'''
	return os.path.join(os.path.dirname(__file__), "mapElitesOutput", str(niches), str(map_num))


def run_experiment_for_map(niches, runs=20, max_iter=40, show_visual=False):
	'''
	Runs the experiment for a map for all leg damage scenario over
	the specified number of runs. Saves all the results to a file.
	:param niches: Number of niches in the map
	:param runs: Number of runs to run the experiment for
	'''
	print(f"Running experiment for map with {niches} niches")

	centroid_path = get_centroids_path(niches)
	map_path = get_map_path(niches, 1)

	# dictionary where key is scenario index and value is the average performance over iterations array.
	average_data = {}
	for scenario_index in range(5):
		print(f"Running experiment for scenario {scenario_index + 1}")
		failures = scenarios[scenario_index]

		# used to store the average performance over all runs for each failed legs combination for the scenario
		all_performance_data = []
		results = []
		for failure_index, failed_legs in enumerate(failures):
			# we want to sum the performance over all combinations of leg damage combinations for each scenario. i.e. for scenario 1, we want to sum the performance over all 6 combinations of leg damage
			performance_sum = np.zeros(max_iter)
			for run in range(runs):
				print(f"Run: {run + 1}, Failed legs: {failed_legs}")

				num_it, best_index, best_perf, new_map, performance_data = MBOA(map_path, centroid_path, failed_legs, max_iter=max_iter, print_output=False, show_visual=show_visual)

				iter_performance = performance_data['performance']
				# add the performances across each iteration to the performance sum
				performance_sum[:len(iter_performance)] += iter_performance

				results.append({
					"run_num": run,
					"leg_damage_scenario_index": scenario_index,
					"num_iterations": num_it,
					"best_index": best_index,
					"best_performance": best_perf
				})
				print(f"Run {run + 1} complete.\nBest index: {best_index}, Best performance: {best_perf}")
				print("--------------------------------------------------")
			
			average_performance = performance_sum / runs
			all_performance_data.append(average_performance)

		# get the average performance for all failed leg combinations for scenario. 
		average_performance = np.mean(all_performance_data, axis=0)
		average_data[scenario_index] = average_performance

		# checkpoint results. 
		# save_results_to_csv(results, niches)

	# save the average performance data to a CSV file
	save_average_performance_to_csv(average_data, niches)


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Error: Incorrect number of arguments.\n" +
		  "Usage: python run_adapt_tests.py <num_runs> <SHOW_VISUAL (1 or 0)>")
		exit()

	runs = int(sys.argv[1])
	SHOW_VISUAL = bool(int(sys.argv[2]))

	for niche in niches:
		run_experiment_for_map(niche, runs=runs, max_iter=40, show_visual=SHOW_VISUAL)

	

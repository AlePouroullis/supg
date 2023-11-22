import pandas as pd
import matplotlib.pyplot as plt

niche_sizes = [5000, 10000, 20000, 40000]
scenarios = [0, 1, 2, 3, 4]  # 0 = no damage, 1 = 1 damaged leg, 2 = two failed legs separated by two functional legs, etc.
colors = ['blue', 'green', 'red', 'purple']  # Different color for each niche size

def visualize_performance_across_niches():
   plot_data = []

   # Read and process data for each niche size
   for size, color in zip(niche_sizes, colors):
      file_path = f'adapt_output/results_{size}.csv'
      data = pd.read_csv(file_path)

      # Average performance for each leg damage scenario
      avg_performance = data.groupby('leg_damage_scenario_index')['best_performance'].mean()
      
      # Store the data for plotting
      for scenario, performance in avg_performance.items():
         plot_data.append((scenario, performance, size, color))

   # Prepare the plot
   plt.figure(figsize=(10, 6))
   for scenario, performance, size, color in plot_data:
      plt.scatter(scenario, performance, color=color, label=f'{size} niches' if scenario == 0 else "")

   # Format x-axis to display integer values
   plt.xticks(range(len(scenarios)), [str(int(scenario)) for scenario in scenarios], fontsize=12)
   plt.yticks(fontsize=12)

   plt.xlabel('Leg Failure Scenario', fontsize=14)
   plt.ylabel('Average Performance',fontsize=14)
   plt.title('Comparison of Performance Across Niche Sizes for Each Failed Leg Scenario',fontsize=16)
   plt.legend(fontsize=12)
   plt.show()

# Function to apply smoothing using rolling average
def smooth_data(data, window_size=5):
    return data.rolling(window=window_size, min_periods=1).mean()

def visualize_num_iterations_across_niches():
      # Function to determine the last iteration
   def get_last_iteration(data):
      non_zero_data = data[data['average_performance'] > 0]
      return non_zero_data['iteration'].max() if not non_zero_data.empty else 0

   # Plot data for each scenario
   plt.figure(figsize=(10, 6))

   for size, color in zip(niche_sizes, colors):
      last_iterations = []

      for scenario in scenarios:
         file_path = f'adapt_output/average_performance_niches_{size}_scenario_{scenario}.csv'
         data = pd.read_csv(file_path)

         # Determine the last iteration for the scenario
         last_iteration = get_last_iteration(data)
         last_iterations.append(last_iteration)
      # Format x-axis to display integer values
      plt.xticks(range(len(scenarios)), [str(int(scenario)) for scenario in scenarios], fontsize=12)
      plt.yticks(fontsize=12)
      # Plotting the last iteration for each scenario
      plt.scatter(scenarios, last_iterations, color=color, label=f'{size} niches')

   plt.xlabel('Leg Failure Scenario', fontsize=14)
   plt.ylabel('Iterations to End of Evaluation', fontsize=14)
   plt.title('Iterations to End of Evaluation for Each Scenario Across Niche Sizes', fontsize=16)
   # make legend font bigger
   plt.legend(fontsize=12)
   plt.show()




if __name__=="__main__":
   # visualize_performance_across_niches()
   visualize_num_iterations_across_niches()
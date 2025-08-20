from ContractOptimizer import ContractOptimizer
import matplotlib.pyplot as plt
import copy
import numpy as np

class LiabilityCostComparison:
    """
    Compares contract outcomes under different provider cost scenarios.
    """
    def __init__(self, baseline_optimizer, cost_scenarios):
        """
        Initializes the comparison tool.
        
        Args:
            baseline_optimizer (ContractOptimizer): A solved optimizer instance.
            cost_scenarios (dict): A dict where keys are scenario names and
                                   values are tuples of quadratic cost coefficients (c0, c1, c2).
        """
        if baseline_optimizer.q_optimal is None:
            raise ValueError("The baseline_optimizer must be solved first.")
            
        self.baseline = baseline_optimizer
        self.scenarios = cost_scenarios
        self.results = {} # To store results for each scenario

    def run_comparison(self):
        """
        Runs the optimization for each defined cost scenario.
        """
        print("--- Running Scenario Comparison ---")
        for name, params in self.scenarios.items():
            print(f"Solving for scenario: '{name}'...")
            
            # Create a deep copy to avoid modifying the original optimizer
            temp_optimizer = copy.deepcopy(self.baseline)
            
            # Dynamically override the cost function for this scenario
            def new_cost_function(q, p=params):
                return p[0] + p[1] * q + p[2] * q**2
            
            temp_optimizer._cost_function = new_cost_function
            
            # Solve the problem with the new cost function
            temp_optimizer.solve()
            
            if temp_optimizer.q_optimal is not None:
                self.results[name] = {
                    'q': temp_optimizer.q_optimal,
                    'p': temp_optimizer.p_optimal,
                    'cost_func': new_cost_function
                }
        print("--- Comparison complete ---")

    def plot_comparison(self):
        """
        Plots the QoS, Price, and Provider Utility for all scenarios against the baseline.
        """
        if not self.results:
            print("No comparison results to plot. Please run 'run_comparison' first.")
            return

        x_list = [i + 1 for i in range(self.baseline.K)]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8))

        # --- Plot Baseline Results ---
        # QoS
        ax1.plot(x_list, self.baseline.q_optimal, marker='o', linestyle='-', color='black', label='Baseline QoS')
        # Price
        ax2.plot(x_list, self.baseline.p_optimal, marker='o', linestyle='-', color='black', label='Baseline Price')
        # Provider Utility
        baseline_utility = self.baseline.p_optimal - self.baseline._cost_function(self.baseline.q_optimal)
        ax3.plot(x_list, baseline_utility, marker='o', linestyle='-', color='black', label='Baseline Provider Utility')

        # --- Plot Scenario Results ---
        markers = ['d', 's', '^', 'v']
        colors = ['cornflowerblue', 'salmon', 'mediumseagreen', 'gold']
        
        for i, (name, result) in enumerate(self.results.items()):
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            # QoS
            ax1.plot(x_list, result['q'], marker=marker, linestyle='--', color=color, label=f'QoS ({name})')
            # Price
            ax2.plot(x_list, result['p'], marker=marker, linestyle='--', color=color, label=f'Price ({name})')
            # Provider Utility
            utility = result['p'] - result['cost_func'](result['q'])
            ax3.plot(x_list, utility, marker=marker, linestyle='--', color=color, label=f'Utility ({name})')

        # --- Formatting ---
        ax1.set_ylabel('QoS Value')
        ax1.grid(True)
        ax1.legend(loc='upper left')

        ax2.set_ylabel('Price Value')
        ax2.grid(True)
        ax2.legend(loc='upper left')

        ax3.set_xlabel('User Types')
        ax3.set_ylabel('Provider Utility')
        ax3.grid(True)
        ax3.legend(loc='upper left')
        
        fig.suptitle('Comparison of Optimal Contracts', fontsize=10, weight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

if __name__ == '__main__':
    # --- Data and Parameters from the Paper ---
    service_configurations = {
        "1": [1, 20, 20, 0.12, 12, 1024, 12, 0.1, 8100.0],
        "2": [2, 100, 20, 0.12, 12, 1024, 12, 0.22, 8100.0],
        "3": [3, 100, 100, 0.12, 12, 1024, 12, 0.35, 15800.0],
        "4": [4, 20, 20, 2.7, 32, 2048, 32, 0.35, 15800.0],
        "5": [5, 100, 20, 2.7, 32, 2048, 32, 0.5, 19500.0],
        "6": [6, 100, 100, 2.7, 32, 2048, 32, 0.7, 19500.0],
        "7": [8, 100, 20, 7, 28, 8192, 16, 0.75, 31200.0],
        "8": [9, 100, 100, 7, 28, 8192, 16, 0.9, 31200.0]
    }
    
    # 1. Create and solve the baseline optimizer
    baseline_optimizer = ContractOptimizer(
        confs=service_configurations, 
        K=15, 
        r=2e7, 
        a=5e-4, 
        delta=0.5,
        epsilon=1e-3
    )
    baseline_optimizer.solve()

    # 2. Define the cost scenarios to compare
    # (c0, c1, c2) for the function c0 + c1*q + c2*q^2
    cost_scenarios = {
        "With Liability": (13.94641, -45.45348, 38.93703),
        "High Infrastructure Cost": (20.0, -50.0, 45.0)
    }

    # 3. Instantiate and run the comparison
    comparison_tool = LiabilityCostComparison(baseline_optimizer, cost_scenarios)
    comparison_tool.run_comparison()
    comparison_tool.plot_comparison()
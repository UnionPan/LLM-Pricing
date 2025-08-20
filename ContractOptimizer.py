import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ContractOptimizer:
    """
    Implements the PACT framework to find the optimal contract menu (QoS, Price)
    for agentic AI services.
    """
    def __init__(self, confs, K, r, a, delta, epsilon):
        # --- System and Contract Parameters ---
        self.confs = confs
        self.K = K  # Number of user types
        self.r = r  # Communication rate (bps)
        self.a = a  # Tokenization parameter (s/KB)
        self.delta = delta  # QoS weight for satisfaction
        
        # --- User Type Characteristics ---
        self.user_types = np.array([i * 2.0 for i in range(self.K)])
        self.type_probabilities = np.array([1.0 / self.K] * self.K)
        
        # --- Data and Results ---
        self.qs_data = []
        self.compus_data = []
        self.q_optimal = None
        self.p_optimal = None
        self.EPSILON = epsilon
        
        # --- Prepare data on initialization ---
        self._prepare_service_data()

    def _calculate_qos(self, conf):
        """Calculates latency and QoS for a single service configuration."""
        # Equation (1): Transmission Time
        t_tran = 8000.0 * (conf[1] + conf[2]) / self.r
        # Equation (2): Tokenization Time
        t_tok = self.a * (conf[1] + conf[2])
        # Equation (3): Inference Time
        num_tokens = (conf[1] + conf[2]) / 4.0 # Simplified token mapping
        flops = num_tokens * (2 * conf[3] * 1e9 + 2 * conf[4] * conf[5] * conf[6])
        t_inf = flops / (conf[8] * 1e9)
        
        # Equation (4): Total Latency
        total_latency = t_tran + t_tok + t_inf
        # Equation (6): Final QoS Score
        qos_score = self.delta * conf[7] + (1.0 - self.delta) * (1.0 - total_latency)
        
        return flops, qos_score

    def _prepare_service_data(self):
        """Populates QoS and computation data from the configurations."""
        for key in self.confs:
            compu, q = self._calculate_qos(self.confs[key])
            self.qs_data.append(q)
            self.compus_data.append(compu)
        

    def _cost_function(self, q):
        """Provider's cost to deliver a certain QoS level (fitted function)."""
        return 13.59411 - 47.59992 * q + 41.70379 * q**2

    def _valuation_function(self, q, alpha=1.0):
        """User's monetary valuation of QoS, with diminishing returns."""
        return 1.0 - np.exp(-alpha * q)

    def _define_objective(self):
        """Returns the objective function (negative profit) for the optimizer."""
        def objective(x):
            q, p = x[:self.K], x[self.K:]
            costs = self._cost_function(np.array(q))
            profit = self.type_probabilities * (p - costs)
            return -np.sum(profit)
        return objective

    def _define_constraints(self):
        """Builds and returns the list of all optimization constraints."""
        constraints = []
        
        # IR constraint for the lowest type
        def ir_constraint(x):
            q, p = x[:self.K], x[self.K:]
            return self.user_types[0] * self._valuation_function(q[0]) - p[0]
        constraints.append({'type': 'ineq', 'fun': ir_constraint})
        
        # IC constraints for k > 1
        for k in range(1, self.K):
            def ic_constraint(x, k=k):
                q, p = x[:self.K], x[self.K:]
                utility_k = self.user_types[k] * self._valuation_function(q[k]) - p[k]
                utility_k_as_j = self.user_types[k] * self._valuation_function(q[k-1]) - p[k-1]
                return utility_k - utility_k_as_j
            constraints.append({'type': 'ineq', 'fun': ic_constraint})
        
        # Monotonicity constraints for QoS and Price
        for k in range(1, self.K):
            def monotonic_q(x, k=k):
                return x[k] - x[k - 1] - self.EPSILON # q_k >= q_{k-1} + epsilon
            constraints.append({'type': 'ineq', 'fun': monotonic_q})
            
            def monotonic_p(x, k=k):
                return x[self.K + k] - x[self.K + k - 1] - self.EPSILON # p_k >= p_{k-1} + epsilon
            constraints.append({'type': 'ineq', 'fun': monotonic_p})
            
        return constraints

    def solve(self):
        """Runs the optimization to find the optimal contract menu."""
        objective_func = self._define_objective()
        constraints = self._define_constraints()
        
        # Define bounds and an initial guess for the solver
        bounds = [(0, 1)] * self.K + [(0, None)] * self.K
        q0 = np.linspace(np.min(self.qs_data), np.max(self.qs_data), self.K)
        p0 = self.user_types * self._valuation_function(q0) - 0.1
        x0 = np.concatenate([q0, p0])
        
        # Run the optimizer
        result = minimize(objective_func, x0, bounds=bounds, constraints=constraints)#, method='SLSQP')
        
        if result.success:
            self.q_optimal = result.x[:self.K]
            self.p_optimal = result.x[self.K:]
            print("✅ Optimization successful!")
            print("\nOptimal QoS Levels:\n", np.round(self.q_optimal, 4))
            print("\nOptimal Prices:\n", np.round(self.p_optimal, 4))
        else:
            print("❌ Optimization failed:", result.message)

    def plot_results(self):
        """Visualizes the optimal contract menu and price scaling with user type."""
        if self.q_optimal is None or self.p_optimal is None:
            print("No results to plot. Please run the 'solve' method first.")
            return

        # Create a figure with two subplots side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # --- Plot 1: Optimal Contract Menu (QoS vs. Price) ---
        ax1.plot(self.q_optimal, self.p_optimal, 'o-', color='royalblue', label='Optimal Contract Menu')
        for i in range(self.K):
            ax1.text(self.q_optimal[i], self.p_optimal[i] * 1.01, f'θ$_{{{i}}}$', fontsize=9, ha='center')
        ax1.set_title('Optimal Contract Menu (QoS vs. Price)', fontsize=14)
        ax1.set_xlabel('Quality of Service (QoS)', fontsize=12)
        ax1.set_ylabel('Price (P)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # --- Plot 2: Price vs. User Type ---
        ax2.plot(self.user_types, self.p_optimal, 'o-', color='crimson', label='Price per User Type')
        ax2.set_title('Price vs. User Type', fontsize=14)
        ax2.set_xlabel('User Type (θ)', fontsize=12)
        ax2.set_ylabel('Optimal Price (P)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        
        # Add a main title for the entire figure
        fig.suptitle('Analysis of Optimal PACT Contracts', fontsize=18, weight='bold')
        
        # Adjust layout to prevent titles/labels from overlapping
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
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
    
    # --- Instantiate and Run the Optimizer ---
    optimizer = ContractOptimizer(
        confs=service_configurations, 
        K=15, 
        r=2e7, 
        a=5e-4, 
        delta=0.5,
        epsilon=1e-3
    )
    optimizer.solve()
    optimizer.plot_results()
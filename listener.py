from typing import Union, Dict, Any
import json
import logging
from enum import Enum

# Import our statistical functions from the previous module
from prob_calc_backend import (
    calculate_permutation,
    calculate_combination,
    bayesian_probability,
    binomial_probability,
    plot_binomial_distribution,
    bernoulli_trial,
    bernoulli_process,
    visualize_bernoulli_process,
    geometric_probability,
    plot_geometric_distribution,
    visualize_bayesian_venn,
    plot_normal_distribution,
    calculate_p_value
)

class StatFunction(Enum):
    PERMUTATION = "permutation"
    COMBINATION = "combination"
    BAYESIAN = "bayesian"
    BAYESIAN_VENN = "bayesian_venn"
    BINOMIAL = "binomial"
    BERNOULLI_TRIAL = "bernoulli_trial"
    BERNOULLI_PROCESS = "bernoulli_process"
    GEOMETRIC = "geometric"
    NORMAL_DISTRIBUTION = "normal_distribution"
    P_VALUE = "p_value"

class StatsHandler:
    def __init__(self):
        """Initialize the stats handler with logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM's response into a structured format.
        Expected format: JSON string containing 'function' and 'parameters'
        
        Example LLM response:
        {
            "function": "binomial",
            "parameters": {
                "n": 10,
                "p": 0.5,
                "k": 3
            },
            "plot": true
        }
        """
        try:
            return json.loads(llm_response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            raise ValueError("Invalid JSON format in LLM response")

    def execute_function(self, parsed_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the requested statistical function with provided parameters
        """
        try:
            function_name = parsed_request.get("function", "").lower()
            parameters = parsed_request.get("parameters", {})
            plot_requested = parsed_request.get("plot", False)
            
            # Match function name to enum
            try:
                function_type = StatFunction(function_name)
            except ValueError:
                raise ValueError(f"Unknown function type: {function_name}")
            
            result = None
            plot_result = None
            
            # Execute the appropriate function
            if function_type == StatFunction.PERMUTATION:
                result = calculate_permutation(
                    parameters.get("n"),
                    parameters.get("r")
                )
                
            elif function_type == StatFunction.COMBINATION:
                result = calculate_combination(
                    parameters.get("n"),
                    parameters.get("r")
                )
                
            elif function_type == StatFunction.BAYESIAN:
                result = bayesian_probability(
                    parameters.get("prior"),
                    parameters.get("likelihood"),
                    parameters.get("evidence")
                )
            
            elif function_type == StatFunction.BAYESIAN_VENN:
                visualize_bayesian_venn(
                    parameters.get("prior"),
                    parameters.get("likelihood"),
                    parameters.get("evidence"),
                    parameters.get("total_number_of_events", 1000)
                )
                plot_result = True
                result = bayesian_probability(
                    parameters.get("prior"),
                    parameters.get("likelihood"),
                    parameters.get("evidence")
                )
                
            elif function_type == StatFunction.BINOMIAL:
                result = binomial_probability(
                    parameters.get("n"),
                    parameters.get("p"),
                    parameters.get("k")
                )
                if plot_requested:
                    plot_binomial_distribution(
                        parameters.get("n"),
                        parameters.get("p")
                    )
                    plot_result = True
            
            elif function_type == StatFunction.BERNOULLI_TRIAL:
                result = bernoulli_trial(parameters.get("p"))
            
            elif function_type == StatFunction.BERNOULLI_PROCESS:
                outcomes = bernoulli_process(
                    parameters.get("p"),
                    parameters.get("n")
                )
                result = {
                    "successes": int(outcomes.sum()),
                    "success_rate": float(outcomes.mean())
                }
                if plot_requested:
                    visualize_bernoulli_process(outcomes, parameters.get("p"))
                    plot_result = True
                    
            elif function_type == StatFunction.GEOMETRIC:
                result = geometric_probability(
                    parameters.get("p"),
                    parameters.get("k")
                )
                if plot_requested:
                    plot_geometric_distribution(
                        parameters.get("p"),
                        parameters.get("max_k", 20)
                    )
                    plot_result = True
            
            elif function_type == StatFunction.NORMAL_DISTRIBUTION:
                result = plot_normal_distribution(
                    parameters.get("mean"),
                    parameters.get("std_dev"),
                    parameters.get("value")
                )
                plot_result = True
            
            elif function_type == StatFunction.P_VALUE:
                result = calculate_p_value(
                    parameters.get("mean"),
                    parameters.get("std_dev"),
                    parameters.get("observed_value"),
                    parameters.get("two_tailed", True)
                )
                plot_result = True
            
            return {
                "success": True,
                "result": result,
                "plot_generated": plot_result,
                "function_executed": function_name
            }
            
        except Exception as e:
            self.logger.error(f"Error executing function: {e}")
            return {
                "success": False,
                "error": str(e),
                "function_attempted": function_name
            }

    def handle_request(self, llm_response: str) -> Dict[str, Any]:
        """
        Main handler method that processes the LLM response and returns results
        """
        try:
            parsed_request = self.parse_llm_response(llm_response)
            return self.execute_function(parsed_request)
        except Exception as e:
            self.logger.error(f"Error handling request: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Example usage:
def main():
    handler = StatsHandler()
    
    # Example LLM responses covering all function types
    example_requests = [
        """
        {
            "function": "binomial",
            "parameters": {
                "n": 10,
                "p": 0.5,
                "k": 3
            },
            "plot": true
        }
        """,
        """
        {
            "function": "bayesian_venn",
            "parameters": {
                "prior": 0.3,
                "likelihood": 0.8,
                "evidence": 0.5,
                "total_number_of_events": 1000
            }
        }
        """,
        """
        {
            "function": "bernoulli_process",
            "parameters": {
                "p": 0.3,
                "n": 1000
            },
            "plot": true
        }
        """,
        """
        {
            "function": "normal_distribution",
            "parameters": {
                "mean": 0,
                "std_dev": 1,
                "value": 1.96
            }
        }
        """,
        """
        {
            "function": "p_value",
            "parameters": {
                "mean": 100,
                "std_dev": 15,
                "observed_value": 125,
                "two_tailed": true
            }
        }
        """
    ]
    
    # Process each request
    for request in example_requests:
        result = handler.handle_request(request)
        print(f"Request result: {result}")

main()
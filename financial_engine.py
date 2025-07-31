from typing import Dict, Union, Optional, Tuple, List
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

@dataclass(frozen=True)
class FinancialInputs:
    users: int
    churn_rate: float
    ltv: float
    monthly_expenses: float
    initial_capital: float
    cac: float
    
    def __post_init__(self):
        if any(x < 0 for x in (self.users, self.churn_rate, self.ltv, self.cac)):
            raise ValueError("Negative values not allowed")
        if self.churn_rate > 100:
            raise ValueError("Churn rate cannot exceed 100%")

@dataclass(frozen=True)
class FinancialMetrics:
    mrr: float
    arr: float
    churn_loss: float
    burn_rate: float
    runway: Union[float, str]
    cac: float
    ltv: float
    ltv_cac_ratio: float
    payback_period: Union[float, str]
    growth_efficiency: float
    profitability: float
    break_even_users: float

class FinancialCalculator:
    __slots__ = ('_precision', '_cache_enabled')
    
    def __init__(self, precision: int = 2, cache_enabled: bool = True):
        self._precision = min(max(precision, 0), 8)
        self._cache_enabled = cache_enabled
    
    def _cached_core_metrics(self, inputs: FinancialInputs) -> Tuple[float, float, float, float, float]:
        if not self._cache_enabled:
            return self._calculate_core_metrics(inputs)
        return self._calculate_core_metrics_cached(inputs)
    
    @lru_cache(maxsize=256)
    def _calculate_core_metrics_cached(self, inputs: FinancialInputs) -> Tuple[float, float, float, float, float]:
        churn_decimal = inputs.churn_rate * 0.01
        arpu = inputs.ltv * churn_decimal
        mrr = inputs.users * arpu
        arr = mrr * 12
        churn_loss = mrr * churn_decimal
        burn_rate = inputs.monthly_expenses - mrr
        return mrr, arr, churn_loss, burn_rate, arpu
    
    def _calculate_core_metrics(self, inputs: FinancialInputs) -> Tuple[float, float, float, float, float]:
        churn_decimal = inputs.churn_rate * 0.01
        arpu = inputs.ltv * churn_decimal
        mrr = inputs.users * arpu
        return (
            mrr,
            mrr * 12,
            mrr * churn_decimal,
            inputs.monthly_expenses - mrr,
            arpu
        )
    
    def _calculate_runway(self, burn_rate: float, capital: float) -> Union[float, str]:
        if burn_rate <= 0 or capital <= 0:
            return float('inf') if burn_rate <= 0 else 0.0
        return capital / abs(burn_rate)
    
    def _calculate_advanced_metrics(self, inputs: FinancialInputs, arpu: float) -> Tuple[float, Union[float, str], float, float, float]:
        ltv_cac_ratio = inputs.ltv / inputs.cac if inputs.cac else float('inf')
        payback_period = inputs.cac / arpu if arpu else float('inf')
        growth_efficiency = (inputs.ltv - inputs.cac) / inputs.cac if inputs.cac else float('inf')
        profitability = (inputs.ltv - inputs.cac) / inputs.ltv if inputs.ltv else 0.0
        break_even = inputs.monthly_expenses / arpu if arpu else float('inf')
        return ltv_cac_ratio, payback_period, growth_efficiency, profitability, break_even
    
    def _round_value(self, value: Union[float, str]) -> Union[float, str]:
        if isinstance(value, str) or value in (float('inf'), float('-inf')):
            return "Infinite" if value == float('inf') else value
        try:
            return float(Decimal(str(value)).quantize(
                Decimal('0.' + '0' * self._precision), 
                rounding=ROUND_HALF_UP
            ))
        except (InvalidOperation, ValueError):
            return value
    
    def calculate_metrics(self, data: Dict) -> Dict:
        """Calculate all financial metrics from input data"""
        try:
            inputs = FinancialInputs(
                users=data['users'],
                churn_rate=data['churn'],
                ltv=data['ltv'],
                monthly_expenses=data['monthly_expenses'],
                initial_capital=data['initial_capital'],
                cac=data['cac']
            )
            
            # Get core metrics
            mrr, arr, churn_loss, burn_rate, arpu = self._cached_core_metrics(inputs)
            
            # Calculate runway
            runway = self._calculate_runway(burn_rate, inputs.initial_capital)
            
            # Calculate advanced metrics
            ltv_cac_ratio, payback_period, growth_efficiency, profitability, break_even = self._calculate_advanced_metrics(inputs, arpu)
            
            # Create and return results
            return {
                'mrr': self._round_value(mrr),
                'arr': self._round_value(arr),
                'churn_loss': self._round_value(churn_loss),
                'burn_rate': self._round_value(burn_rate),
                'runway': self._round_value(runway) if isinstance(runway, (int, float)) else runway,
                'cac': self._round_value(inputs.cac),
                'ltv': self._round_value(inputs.ltv),
                'ltv_cac_ratio': self._round_value(ltv_cac_ratio) if isinstance(ltv_cac_ratio, (int, float)) else ltv_cac_ratio,
                'payback_period': self._round_value(payback_period) if isinstance(payback_period, (int, float)) else payback_period,
                'growth_efficiency': self._round_value(growth_efficiency) if isinstance(growth_efficiency, (int, float)) else growth_efficiency,
                'profitability': self._round_value(profitability) if isinstance(profitability, (int, float)) else profitability,
                'break_even_users': self._round_value(break_even) if isinstance(break_even, (int, float)) else break_even
            }
            
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except Exception as e:
            raise ValueError(f"Error calculating metrics: {str(e)}")
    
    def batch_calculate(self, scenarios: List[Dict]) -> List[Dict]:
        return [calculate_metrics(scenario) for scenario in scenarios]
    
    def scenario_analysis(self, base_data: Dict, variations: Dict[str, Dict]) -> Dict[str, Dict]:
        base_result = calculate_metrics(base_data)
        result = {scenario: calculate_metrics({**base_data, **changes}) 
                 for scenario, changes in variations.items()}
        result['base'] = base_result
        return result

# Add these functions at the end of the file, before the if __name__ == '__main__' block

def calculate_metrics(data: Dict) -> Dict:
    """Standalone function to calculate all financial metrics"""
    calculator = FinancialCalculator()
    return calculator.calculate_metrics(data)

def calculate_mrr_arr(data: Dict) -> Dict:
    """Calculate Monthly Recurring Revenue (MRR) and Annual Recurring Revenue (ARR)"""
    calculator = FinancialCalculator()
    metrics = calculator.calculate_metrics(data)
    return {
        'mrr': metrics['mrr'],
        'arr': metrics['arr']
    }

def calculate_burn_rate_runway(data: Dict) -> Dict:
    """Calculate burn rate and runway"""
    calculator = FinancialCalculator()
    metrics = calculator.calculate_metrics(data)
    return {
        'burn_rate': metrics['burn_rate'],
        'runway': metrics['runway']
    }

def calculate_ltv_cac(data: Dict) -> Dict:
    """Calculate LTV, CAC, and related metrics"""
    calculator = FinancialCalculator()
    metrics = calculator.calculate_metrics(data)
    return {
        'ltv': metrics['ltv'],
        'cac': metrics['cac'],
        'ltv_cac_ratio': metrics['ltv_cac_ratio'],
        'payback_period': metrics['payback_period']
    }

if __name__ == '__main__':
    import json
    from timeit import timeit
    
    calc = FinancialCalculator(cache_enabled=True)
    
    test_data = {
        'users': 1000,
        'churn': 5.5,
        'ltv': 2500,
        'monthly_expenses': 50000,
        'initial_capital': 250000,
        'cac': 300
    }
    
    # Performance test
    time_taken = timeit(
        lambda: calc.calculate_metrics(test_data), 
        number=10000
    )
    print(f"Calculation time for 10,000 iterations: {time_taken:.4f} seconds")
    
    # Single calculation
    metrics = calc.calculate_metrics(test_data)
    print(json.dumps(metrics, indent=2))
    
    # Scenario analysis
    scenarios = {
        'high_growth': {'users': 1500, 'churn': 4.0},
        'cost_cutting': {'monthly_expenses': 40000, 'cac': 250},
        'worst_case': {'users': 800, 'churn': 7.0, 'monthly_expenses': 55000}
    }
    
    analysis = calc.scenario_analysis(test_data, scenarios)
    print("\nScenario Analysis:")
    print(json.dumps(analysis, indent=2))
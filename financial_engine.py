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
    monthly_growth_rate: float
    
    def __post_init__(self):
        # Validate all numeric inputs
        if any(x < 0 for x in (self.users, self.churn_rate, self.ltv, self.cac, self.monthly_expenses, self.initial_capital)):
            raise ValueError("Negative values not allowed for financial inputs")
        
        # Validate churn rate is between 0.001 and 0.999 (0.1% to 99.9%)
        if not (0.001 <= self.churn_rate <= 0.999):
            raise ValueError("Churn rate must be between 0.001 and 0.999 (0.1% to 99.9%)")
        
        # Validate CAC is reasonable
        if not (0.01 <= self.cac <= 10000):
            raise ValueError("CAC must be between 0.01 and 10000")
        
        # Validate users is reasonable
        if self.users == 0:
            raise ValueError("Users must be greater than 0")
        
        # Validate LTV is reasonable
        if self.ltv <= 0:
            raise ValueError("LTV must be greater than 0")

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
        """Get core metrics with optional caching"""
        if not self._cache_enabled:
            return self._calculate_core_metrics(inputs)
        return self._calculate_core_metrics_cached(inputs)
    
    @lru_cache(maxsize=256)
    def _calculate_core_metrics_cached(self, inputs: FinancialInputs) -> Tuple[float, float, float, float, float]:
        """Cached version of core metrics calculation"""
        return self._calculate_core_metrics(inputs)
    
    def _calculate_core_metrics(self, inputs: FinancialInputs) -> Tuple[float, float, float, float, float]:
        """Calculate core financial metrics"""
        # ARPU calculation: LTV * churn_rate (assuming churn_rate is already decimal)
        arpu = inputs.ltv * inputs.churn_rate
        
        # MRR calculation
        mrr = inputs.users * arpu
        
        # ARR calculation
        arr = mrr * 12
        
        # Monthly churn loss
        churn_loss = mrr * inputs.churn_rate
        
        # Burn rate (negative means profitable)
        burn_rate = inputs.monthly_expenses - mrr
        
        return mrr, arr, churn_loss, burn_rate, arpu
    
    def _calculate_runway(self, burn_rate: float, capital: float) -> Union[float, str]:
        """Calculate runway in months"""
        if burn_rate <= 0:
            return "Infinite"  # Profitable or break-even
        if capital <= 0:
            return 0.0
        return capital / burn_rate
    
    def generate_projections(self, inputs: FinancialInputs, arpu: float) -> Dict[str, List]:
        """Generate 12-month financial projections."""
        mrr_projection = []
        capital_projection = []
        months = [f"Month {i+1}" for i in range(12)]

        current_users = float(inputs.users)
        current_capital = float(inputs.initial_capital)
        growth_rate = inputs.monthly_growth_rate / 100.0

        for _ in range(12):
            mrr = current_users * arpu
            mrr_projection.append(self._round_value(mrr))

            net_flow = mrr - inputs.monthly_expenses
            current_capital += net_flow
            capital_projection.append(self._round_value(max(0, current_capital)))

            churned_users = current_users * inputs.churn_rate
            new_users = current_users * growth_rate
            current_users += new_users - churned_users
            current_users = max(0, current_users)

        return {
            "months": months,
            "mrr_projection": mrr_projection,
            "capital_projection": capital_projection,
        }

    def _calculate_advanced_metrics(self, inputs: FinancialInputs, arpu: float) -> Tuple[float, Union[float, str], float, float, float]:
        """Calculate advanced business metrics"""
        # LTV:CAC ratio
        ltv_cac_ratio = inputs.ltv / inputs.cac if inputs.cac > 0 else float('inf')
        
        # Payback period in months
        payback_period = inputs.cac / arpu if arpu > 0 else "Infinite"
        
        # Growth efficiency
        growth_efficiency = (inputs.ltv - inputs.cac) / inputs.cac if inputs.cac > 0 else float('inf')
        
        # Profitability percentage
        profitability = (inputs.ltv - inputs.cac) / inputs.ltv if inputs.ltv > 0 else 0.0
        
        # Break-even users needed
        break_even = inputs.monthly_expenses / arpu if arpu > 0 else float('inf')
        
        return ltv_cac_ratio, payback_period, growth_efficiency, profitability, break_even
    
    def _round_value(self, value: Union[float, str]) -> Union[float, str]:
        """Round numerical values to specified precision"""
        if isinstance(value, str):
            return value
        if value in (float('inf'), float('-inf')):
            return "Infinite"
        if np.isnan(value):
            return "N/A"
        
        try:
            return float(Decimal(str(value)).quantize(
                Decimal('0.' + '0' * self._precision), 
                rounding=ROUND_HALF_UP
            ))
        except (InvalidOperation, ValueError, OverflowError):
            return value
    
    def calculate_metrics(self, data: Dict) -> Dict:
        """Calculate all financial metrics from input data"""
        try:
            # Create validated inputs
            inputs = FinancialInputs(
                users=int(data['users']),
                churn_rate=float(data['churn']),
                ltv=float(data['ltv']),
                monthly_expenses=float(data['monthly_expenses']),
                initial_capital=float(data['initial_capital']),
                cac=float(data['cac']),
                monthly_growth_rate=float(data.get('monthly_growth', 0))
            )
            
            # Get core metrics
            mrr, arr, churn_loss, burn_rate, arpu = self._cached_core_metrics(inputs)
            
            # Calculate runway
            runway = self._calculate_runway(burn_rate, inputs.initial_capital)
            
            # Calculate advanced metrics
            ltv_cac_ratio, payback_period, growth_efficiency, profitability, break_even = self._calculate_advanced_metrics(inputs, arpu)

            # Generate projections
            projections = self.generate_projections(inputs, arpu)
            
            # Create and return results with proper rounding
            return {
                'mrr': self._round_value(mrr),
                'arr': self._round_value(arr),
                'churn_loss': self._round_value(churn_loss),
                'burn_rate': self._round_value(burn_rate),
                'runway': self._round_value(runway) if isinstance(runway, (int, float)) else runway,
                'cac': self._round_value(inputs.cac),
                'ltv': self._round_value(inputs.ltv),
                'ltv_cac_ratio': self._round_value(ltv_cac_ratio) if isinstance(ltv_cac_ratio, (int, float)) else "Infinite",
                'payback_period': self._round_value(payback_period) if isinstance(payback_period, (int, float)) else payback_period,
                'growth_efficiency': self._round_value(growth_efficiency) if isinstance(growth_efficiency, (int, float)) else "Infinite",
                'profitability': self._round_value(profitability),
                'break_even_users': self._round_value(break_even) if isinstance(break_even, (int, float)) else "Infinite",
                'projections': projections
            }
            
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input data: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error calculating metrics: {str(e)}")
    
    def batch_calculate(self, scenarios: List[Dict]) -> List[Dict]:
        """Calculate metrics for multiple scenarios"""
        return [self.calculate_metrics(scenario) for scenario in scenarios]
    
    def scenario_analysis(self, base_data: Dict, variations: Dict[str, Dict]) -> Dict[str, Dict]:
        """Perform scenario analysis with variations"""
        base_result = self.calculate_metrics(base_data)
        result = {scenario: self.calculate_metrics({**base_data, **changes}) 
                 for scenario, changes in variations.items()}
        result['base'] = base_result
        return result

# Standalone utility functions
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

# Input validation helper function
def validate_financial_inputs(data: Dict) -> Dict[str, List[str]]:
    """Validate financial inputs and return validation errors"""
    errors = {}
    
    try:
        cac = float(data.get('cac', 0))
        if not (0.01 <= cac <= 10000):
            errors['cac'] = ["Must be greater than or equal to 0.01 and less than or equal to 10000."]
    except (ValueError, TypeError):
        errors['cac'] = ["Must be a valid number."]
    
    try:
        churn = float(data.get('churn', 0))
        if not (0.001 <= churn <= 0.999):
            errors['churn'] = ["Must be greater than or equal to 0.001 and less than or equal to 0.999."]
    except (ValueError, TypeError):
        errors['churn'] = ["Must be a valid number."]
    
    try:
        users = int(data.get('users', 0))
        if users <= 0:
            errors['users'] = ["Must be greater than 0."]
    except (ValueError, TypeError):
        errors['users'] = ["Must be a valid integer."]
    
    try:
        ltv = float(data.get('ltv', 0))
        if ltv <= 0:
            errors['ltv'] = ["Must be greater than 0."]
    except (ValueError, TypeError):
        errors['ltv'] = ["Must be a valid number."]
    
    try:
        monthly_expenses = float(data.get('monthly_expenses', 0))
        if monthly_expenses < 0:
            errors['monthly_expenses'] = ["Must be greater than or equal to 0."]
    except (ValueError, TypeError):
        errors['monthly_expenses'] = ["Must be a valid number."]
    
    try:
        initial_capital = float(data.get('initial_capital', 0))
        if initial_capital < 0:
            errors['initial_capital'] = ["Must be greater than or equal to 0."]
    except (ValueError, TypeError):
        errors['initial_capital'] = ["Must be a valid number."]
    
    return errors

if __name__ == '__main__':
    import json
    from timeit import timeit
    
    # Test with valid data
    test_data = {
        'users': 1000,
        'churn': 0.055,  # 5.5% as decimal
        'ltv': 2500,
        'monthly_expenses': 50000,
        'initial_capital': 250000,
        'cac': 300
    }
    
    # Validate inputs first
    validation_errors = validate_financial_inputs(test_data)
    if validation_errors:
        print("❌ Validation Errors:")
        print(json.dumps(validation_errors, indent=2))
    else:
        print("✅ All inputs are valid!")
        
        calc = FinancialCalculator(cache_enabled=True)
        
        # Performance test
        time_taken = timeit(
            lambda: calc.calculate_metrics(test_data), 
            number=10000
        )
        print(f"⚡ Calculation time for 10,000 iterations: {time_taken:.4f} seconds")
        
        # Single calculation
        metrics = calc.calculate_metrics(test_data)
        print("\n📊 Financial Metrics:")
        print(json.dumps(metrics, indent=2))
        
        # Scenario analysis
        scenarios = {
            'high_growth': {'users': 1500, 'churn': 0.04},  # 4% churn
            'cost_cutting': {'monthly_expenses': 40000, 'cac': 250},
            'worst_case': {'users': 800, 'churn': 0.07, 'monthly_expenses': 55000}  # 7% churn
        }
        
        analysis = calc.scenario_analysis(test_data, scenarios)
        print("\n🔍 Scenario Analysis:")
        print(json.dumps(analysis, indent=2))
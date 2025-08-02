import json
import os
import time
from urllib.parse import parse_qs

def calculate_metrics(data):
    """Comprehensive financial metrics calculation"""
    # Extract input values
    users = data.get('users', 0)
    ltv = data.get('ltv', 0)  # ARPU
    cac = data.get('cac', 0)
    churn = data.get('churn', 0.05)
    monthly_expenses = data.get('monthly_expenses', 0)
    initial_capital = data.get('initial_capital', 0)
    monthly_growth = data.get('monthly_growth', 0) / 100  # Convert percentage to decimal
    cogs_percent = data.get('cogs', 20) / 100  # Convert percentage to decimal
    
    # Calculate core metrics
    mrr = users * ltv  # Monthly Recurring Revenue
    arr = mrr * 12  # Annual Recurring Revenue
    
    # Calculate LTV (actual lifetime value, not just ARPU)
    actual_ltv = ltv / max(churn, 0.001)  # LTV = ARPU / Churn Rate
    
    # LTV/CAC ratio
    ltv_cac_ratio = actual_ltv / max(cac, 1)
    
    # Burn rate and runway
    burn_rate = monthly_expenses
    runway = initial_capital / max(burn_rate, 1) if burn_rate > 0 else float('inf')
    
    # Break even users
    break_even_users = monthly_expenses / max(ltv * (1 - cogs_percent), 1)
    
    # Churn loss
    churn_loss = mrr * churn
    
    # Growth efficiency and potential
    growth_efficiency = (monthly_growth * 100) / max(cac / ltv, 1) if cac > 0 else 0
    growth_potential = min(100, (ltv_cac_ratio / 3) * 50 + (1 - churn) * 50)
    
    # Payback period (in months)
    payback_period = cac / max(ltv * (1 - cogs_percent), 1)
    
    # Profitability
    gross_profit_margin = 1 - cogs_percent
    profitability = (mrr * gross_profit_margin - monthly_expenses) / max(mrr, 1)
    
    # Risk score calculation
    risk_score = min(100, (churn * 40) + (min(1, cac/actual_ltv) * 30) + (min(1, burn_rate/max(initial_capital, 1)) * 30))
    
    # Predicted revenue (12-month projection)
    predicted_revenue = mrr * 12 * (1 + monthly_growth) ** 12
    
    # Generate 12-month projections
    months = [f"Month {i+1}" for i in range(12)]
    mrr_projection = []
    capital_projection = []
    current_mrr = mrr
    current_capital = initial_capital
    
    for month in range(12):
        # Apply growth and churn
        current_mrr = current_mrr * (1 + monthly_growth - churn)
        current_capital = max(0, current_capital - burn_rate)
        
        mrr_projection.append(round(current_mrr, 2))
        capital_projection.append(round(current_capital, 2))
    
    return {
        # Core metrics
        "mrr": round(mrr, 2),
        "arr": round(arr, 2),
        "ltv": round(actual_ltv, 2),
        "cac": round(cac, 2),
        "ltv_cac_ratio": round(ltv_cac_ratio, 2),
        "burn_rate": round(burn_rate, 2),
        "runway": round(runway, 2) if runway != float('inf') else "Infinite",
        "break_even_users": round(break_even_users, 2),
        "churn_loss": round(churn_loss, 2),
        "growth_efficiency": round(growth_efficiency, 2),
        "growth_potential": round(growth_potential, 2),
        "payback_period": round(payback_period, 2),
        "profitability": round(profitability, 4),
        "risk_score": round(risk_score, 2),
        "predicted_revenue": round(predicted_revenue, 2),
        
        # Projections for charts
        "projections": {
            "months": months,
            "mrr_projection": mrr_projection,
            "capital_projection": capital_projection
        }
    }

def handler(event, context):
    """Netlify serverless function handler"""
    
    # Enable CORS
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Content-Type': 'application/json'
    }
    
    # Handle preflight OPTIONS request
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    try:
        # Parse the path to determine the endpoint
        path = event.get('path', '').replace('/.netlify/functions/api', '')
        method = event.get('httpMethod', 'GET')
        
        # Health check endpoint
        if path == '/health' and method == 'GET':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'status': 'healthy',
                    'timestamp': int(time.time()),
                    'version': '2.0.0',
                    'platform': 'netlify'
                })
            }
        
        # Main calculation endpoint
        if path == '/v2/calculate' and method == 'POST':
            # Parse request body
            body = event.get('body', '{}')
            if isinstance(body, str):
                request_data = json.loads(body)
            else:
                request_data = body
            
            # Validate required fields
            required_fields = ['users', 'ltv', 'cac', 'churn', 'monthly_expenses', 'initial_capital']
            for field in required_fields:
                if field not in request_data:
                    return {
                        'statusCode': 400,
                        'headers': headers,
                        'body': json.dumps({
                            'success': False,
                            'error': 'Missing required field',
                            'field': field
                        })
                    }
            
            # Calculate metrics
            results = calculate_metrics(request_data)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'success': True,
                    'data': {'summary_metrics': results},
                    'cached': False,
                    'processing_time': '0.1s'
                })
            }
        
        # Test endpoint
        if path == '/test' and method == 'GET':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'success': True,
                    'message': 'Netlify function working',
                    'timestamp': int(time.time())
                })
            }
        
        # Default 404 response
        return {
            'statusCode': 404,
            'headers': headers,
            'body': json.dumps({
                'success': False,
                'error': 'Endpoint not found',
                'path': path,
                'method': method
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'success': False,
                'error': 'Internal server error',
                'message': str(e)
            })
        }

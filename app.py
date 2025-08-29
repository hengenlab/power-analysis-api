# Flask Power Analysis API
# Deploy this to Render, Heroku, or similar service

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy import stats
from scipy.optimize import brentq
import math

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from your Squarespace site

def nct_cdf(x, df, nc):
    """
    Non-central t cumulative distribution function.
    More accurate than JavaScript implementation.
    """
    if nc == 0:
        return stats.t.cdf(x, df)
    
    # Use scipy's implementation
    return stats.nct.cdf(x, df, nc)

def calculate_power(n, d, alpha, test_type, alternative):
    """Calculate statistical power given sample size and effect size."""
    
    if test_type == 'two.sample':
        df = 2 * (n - 1)
        ncp = d * math.sqrt(n / 2)
    elif test_type == 'one.sample' or test_type == 'paired':
        df = n - 1  
        ncp = d * math.sqrt(n)
    
    if alternative == 'two.sided':
        t_crit = stats.t.ppf(1 - alpha/2, df)
        power = 1 - nct_cdf(t_crit, df, ncp) + nct_cdf(-t_crit, df, ncp)
    elif alternative == 'greater':
        t_crit = stats.t.ppf(1 - alpha, df)
        power = 1 - nct_cdf(t_crit, df, ncp)
    else:  # less
        t_crit = stats.t.ppf(alpha, df)
        power = nct_cdf(t_crit, df, ncp)
    
    return max(0, min(1, power))  # Clamp to [0,1]

def calculate_sample_size(d, alpha, power, test_type, alternative):
    """Calculate required sample size given effect size and desired power."""
    
    if abs(d) < 0.001:
        raise ValueError("Effect size too small")
    
    def power_function(n):
        if n < 2:
            return -1
        try:
            return calculate_power(n, d, alpha, test_type, alternative) - power
        except:
            return -1
    
    # Find reasonable bounds
    n_low = 2
    n_high = 10
    
    # Expand upper bound until we bracket the root
    while power_function(n_high) < 0 and n_high < 10000:
        n_high *= 2
    
    if power_function(n_high) < 0:
        raise ValueError("Cannot achieve desired power with reasonable sample size")
    
    # Use Brent's method to find root
    n_optimal = brentq(power_function, n_low, n_high, xtol=1e-6)
    return n_optimal

def calculate_effect_size(n, alpha, power, test_type, alternative):
    """Calculate required effect size given sample size and desired power."""
    
    def effect_function(d):
        try:
            return calculate_power(n, d, alpha, test_type, alternative) - power
        except:
            return -1
    
    # Search for effect size
    try:
        d_optimal = brentq(effect_function, 0.001, 5.0, xtol=1e-6)
        return d_optimal
    except ValueError:
        # Try negative direction
        try:
            d_optimal = brentq(effect_function, -5.0, -0.001, xtol=1e-6)  
            return d_optimal
        except ValueError:
            raise ValueError("Cannot find effect size for given parameters")

@app.route('/power-analysis', methods=['POST'])
def power_analysis():
    """Main power analysis endpoint."""
    
    try:
        data = request.json
        
        # Extract parameters
        n = data.get('n')
        d = data.get('d') 
        alpha = data.get('alpha', 0.05)
        power = data.get('power')
        test_type = data.get('test_type', 'two.sample')
        alternative = data.get('alternative', 'two.sided')
        
        # Convert string numbers to float, handle empty strings
        def safe_float(val):
            if val == '' or val is None:
                return None
            return float(val)
        
        n = safe_float(n)
        d = safe_float(d)
        alpha = safe_float(alpha)
        power = safe_float(power)
        
        # Count missing parameters
        params = [n, d, alpha, power]
        missing_count = sum(1 for p in params if p is None)
        
        if missing_count != 1:
            return jsonify({
                'error': 'Please provide exactly 3 parameters and leave 1 empty to calculate'
            }), 400
        
        # Input validation
        if alpha is not None and (alpha <= 0 or alpha >= 1):
            return jsonify({'error': 'Alpha must be between 0 and 1'}), 400
            
        if power is not None and (power <= 0 or power >= 1):
            return jsonify({'error': 'Power must be between 0 and 1'}), 400
            
        if n is not None and n < 2:
            return jsonify({'error': 'Sample size must be at least 2'}), 400
        
        # Perform calculation based on missing parameter
        result = {
            'test_type': test_type,
            'alternative': alternative,
            'alpha': alpha,
        }
        
        if n is None:
            # Calculate sample size
            calculated_n = calculate_sample_size(d, alpha, power, test_type, alternative)
            result.update({
                'n': round(calculated_n, 4),
                'n_rounded': math.ceil(calculated_n),
                'd': d,
                'power': power,
                'calculated': 'sample_size'
            })
            
        elif power is None:
            # Calculate power
            calculated_power = calculate_power(n, d, alpha, test_type, alternative)
            result.update({
                'n': n,
                'd': d, 
                'power': round(calculated_power, 6),
                'power_percent': round(calculated_power * 100, 2),
                'calculated': 'power'
            })
            
        elif d is None:
            # Calculate effect size
            calculated_d = calculate_effect_size(n, alpha, power, test_type, alternative)
            result.update({
                'n': n,
                'd': round(calculated_d, 6),
                'power': power,
                'calculated': 'effect_size'
            })
            
        elif alpha is None:
            # Calculate alpha - more complex, not commonly needed
            return jsonify({'error': 'Alpha calculation not implemented'}), 400
            
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500

@app.route('/cohens-d', methods=['POST'])
def cohens_d():
    """Calculate Cohen's d from means and standard deviations."""
    
    try:
        data = request.json
        
        m1 = float(data['m1'])
        m2 = float(data['m2'])
        s1 = float(data['s1']) 
        s2 = float(data['s2'])
        
        # Calculate pooled standard deviation
        pooled_sd = math.sqrt((s1**2 + s2**2) / 2)
        
        # Calculate Cohen's d
        cohens_d = (m1 - m2) / pooled_sd
        
        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'  
        else:
            interpretation = 'large'
            
        return jsonify({
            'cohens_d': round(cohens_d, 6),
            'abs_cohens_d': round(abs_d, 6),
            'interpretation': interpretation,
            'pooled_sd': round(pooled_sd, 6)
        })
        
    except (ValueError, KeyError) as e:
        return jsonify({'error': 'Please provide valid numeric values for all parameters'}), 400
    except Exception as e:
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'power-analysis-api'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
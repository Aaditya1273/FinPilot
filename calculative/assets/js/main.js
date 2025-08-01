class FundPilot {
    constructor() {
        this.API_ENDPOINT = '/api/calculate'; // Ensure this matches your Flask endpoint
        this.elements = {
            form: document.getElementById('financial-form'),
            spinner: document.getElementById('spinner'),
            resultsSection: document.getElementById('results-section'),
            metricsGrid: document.getElementById('metrics-grid'),
            chartsSection: document.getElementById('charts-section'),
            recommendationsSection: document.getElementById('recommendations-section'),
            recommendationsContent: document.getElementById('recommendations-content'),
            notification: document.getElementById('notification'),
            exportJsonButton: document.getElementById('export-json'),
            exportCsvButton: document.getElementById('export-csv'),
            exportPdfButton: document.getElementById('export-pdf'),
        };
        this.chartInstances = {};
        this.lastData = null;

        if (!this.elements.form) {
            console.error('Financial form not found!');
            return;
        }

        this.init();
    }

    init() {
        this.elements.form.addEventListener('submit', this.handleSubmit.bind(this));
        this.elements.exportJsonButton.addEventListener('click', () => this.exportData('json'));
        this.elements.exportCsvButton.addEventListener('click', () => this.exportData('csv'));
        this.elements.exportPdfButton.addEventListener('click', this.exportPdf.bind(this));
    }

    async handleSubmit(event) {
        event.preventDefault();
        this.toggleSpinner(true);

        const formData = new FormData(this.elements.form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch(this.API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (!response.ok) {
                const errorDetails = result.details ? JSON.stringify(result.details) : result.error;
                throw new Error(errorDetails || `HTTP error! status: ${response.status}`);
            }
            this.lastData = result.data;
            this.updateUI(result.data);
            this.elements.resultsSection.style.display = 'block';

        } catch (error) {
            console.error('Error:', error);
            this.handleApiError(error);
        } finally {
            this.toggleSpinner(false);
        }
    }

    updateUI(data) {
        this.elements.resultsSection.style.display = 'block';
        this.elements.metricsGrid.innerHTML = ''; // Clear previous results

        const metricsToShow = {
            arr: { title: 'Arr', type: 'currency' },
            break_even_users: { title: 'Break Even Users', type: 'number' },
            burn_rate: { title: 'Burn Rate', type: 'currency' },
            cac: { title: 'Cac', type: 'currency' },
            churn_loss: { title: 'Churn Loss', type: 'currency' },
            growth_efficiency: { title: 'Growth Efficiency', type: 'number' },
            growth_potential: { title: 'Growth Potential', type: 'number' },
            ltv: { title: 'Ltv', type: 'currency' },
            ltv_cac_ratio: { title: 'Ltv Cac Ratio', type: 'number' },
            mrr: { title: 'Mrr', type: 'currency' },
            payback_period: { title: 'Payback Period', type: 'currency' },
            predicted_revenue: { title: 'Predicted Revenue', type: 'currency' },
            profitability: { title: 'Profitability', type: 'currency' },
            risk_score: { title: 'Risk Score', type: 'number' },
            runway: { title: 'Runway', type: 'number' },
        };

        for (const key in metricsToShow) {
            if (data.summary_metrics.hasOwnProperty(key)) {
                const metric = metricsToShow[key];
                const value = data.summary_metrics[key];
                const card = `
                    <div class="metric-card">
                        <div class="metric-title">${metric.title}</div>
                        <div class="metric-value">${this.formatMetric(value, metric.type)}</div>
                    </div>
                `;
                this.elements.metricsGrid.innerHTML += card;
            }
        }

        if (data.recommendations && data.recommendations.length > 0) {
            this.elements.recommendationsSection.style.display = 'block';
            this.elements.recommendationsContent.innerHTML = data.recommendations.map(rec => `<div class="recommendation-card">${rec}</div>`).join('');
        } else {
            this.elements.recommendationsSection.style.display = 'none';
        }

        if (data.summary_metrics.projections) {
            this.elements.chartsSection.style.display = 'block';
            this.renderCharts(data.summary_metrics);
        }
    }

    handleApiError(error) {
        this.showNotification(`API Error: ${error.message}`, 'error');
    }

    toggleSpinner(show) {
        if (this.elements.spinner) this.elements.spinner.style.display = show ? 'block' : 'none';
    }

    renderCharts(metrics) {
        const { projections, ltv, cac } = metrics;

        this.renderPieChart('unit-economics-chart', 'Unit Economics', ['LTV', 'CAC'], [ltv, cac]);
        this.renderLineChart('runway-chart', 'Runway & Capital', projections.months, 
            [
                { label: 'Capital ($)', data: projections.capital_projection, borderColor: 'rgba(75, 192, 192, 1)' },
            ]);
        this.renderLineChart('mrr-chart-canvas', 'MRR Growth', projections.months, 
            [
                { label: 'MRR ($)', data: projections.mrr_projection, borderColor: 'rgba(153, 102, 255, 1)' },
            ]);
    }

    renderPieChart(canvasId, label, labels, data) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        if (this.chartInstances[canvasId]) {
            this.chartInstances[canvasId].destroy();
        }
        this.chartInstances[canvasId] = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: data,
                    backgroundColor: ['rgba(54, 162, 235, 0.8)', 'rgba(255, 99, 132, 0.8)'],
                    borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)'],
                    borderWidth: 1
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    renderLineChart(canvasId, label, labels, datasets) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        if (this.chartInstances[canvasId]) {
            this.chartInstances[canvasId].destroy();
        }
        this.chartInstances[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }

    showNotification(message, type = 'info') {
        this.elements.notification.textContent = message;
        this.elements.notification.className = `notification ${type}`;
        this.elements.notification.style.display = 'block';
        setTimeout(() => {
            this.elements.notification.style.display = 'none';
        }, 5000);
    }

    formatMetricName(name) {
        return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    formatMetric(value, type) {
        if (typeof value !== 'number') {
            return value;
        }

        if (type === 'currency') {
            if (Math.abs(value) >= 1000000) {
                return '$' + (value / 1000000).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + 'M';
            } else if (Math.abs(value) >= 1000) {
                return '$' + (value / 1000).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + 'K';
            } else {
                return value.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
            }
        } else { // type is 'number'
            if (Math.abs(value) >= 1000000) {
                return (value / 1000000).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + 'M';
            } else if (Math.abs(value) >= 1000) {
                return (value / 1000).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + 'K';
            } else {
                return value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            }
        }
    }

    exportData(format) {
        if (!this.lastData) {
            this.showNotification('No data to export.', 'error');
            return;
        }

        let dataStr;
        let fileName = `financial_metrics_${new Date().toISOString()}`;

        if (format === 'json') {
            dataStr = JSON.stringify(this.lastData, null, 2);
            fileName += '.json';
        } else { // csv
            const headers = Object.keys(this.lastData.summary_metrics);
            const rows = [headers.join(',')];
            const values = headers.map(header => this.lastData.summary_metrics[header]);
            rows.push(values.join(','));
            dataStr = rows.join('\n');
            fileName += '.csv';
        }

        const blob = new Blob([dataStr], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    exportPdf() {
        if (!this.lastData) {
            this.showNotification('No data to generate PDF.', 'error');
            return;
        }
        const { resultsSection } = this.elements;
        const options = {
            margin: 0.5,
            filename: `financial_report_${new Date().toISOString()}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
        };
        html2pdf().from(resultsSection).set(options).save();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.fundPilotApp = new FundPilot();
});
// Add this to your HTML or existing JS file

class FundPilotAPI {
    constructor(baseUrl = 'http://127.0.0.1:5000') {
        this.baseUrl = baseUrl;
        this.apiKey = 'test-key-123'; // Use your actual API key
    }

    /**
     * Make a safe API request with comprehensive error handling
     */
    async makeRequest(endpoint, data = null, method = 'GET') {
        const url = `${this.baseUrl}${endpoint}`;
        console.log(`📡 Making ${method} request to:`, url);
        console.log('📦 Request data:', data);

        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-API-Key': this.apiKey
            }
        };

        if (data && (method === 'POST' || method === 'PUT')) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, options);
            
            // Get content type to determine how to parse response
            const contentType = response.headers.get('content-type');
            console.log(`📋 Response status: ${response.status}`);
            console.log(`📋 Content-Type: ${contentType}`);

            // Check if response is OK
            if (!response.ok) {
                let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                
                try {
                    // Try to get error details from response
                    if (contentType && contentType.includes('application/json')) {
                        const errorData = await response.json();
                        errorMessage = errorData.message || errorData.error || errorMessage;
                        console.error('❌ API Error Details:', errorData);
                    } else {
                        const errorText = await response.text();
                        console.error('❌ Non-JSON Error Response:', errorText.substring(0, 200));
                        errorMessage = `Server returned HTML instead of JSON. Status: ${response.status}`;
                    }
                } catch (parseError) {
                    console.error('❌ Failed to parse error response:', parseError);
                }
                
                throw new Error(errorMessage);
            }

            // Parse successful response
            if (contentType && contentType.includes('application/json')) {
                const jsonData = await response.json();
                console.log('✅ Success Response:', jsonData);
                return jsonData;
            } else {
                const textData = await response.text();
                console.warn('⚠️ Expected JSON but got text:', textData.substring(0, 100));
                throw new Error('Server returned non-JSON response when JSON was expected');
            }

        } catch (error) {
            console.error('❌ Request Failed:', error.message);
            
            // Check for network errors
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Network error: Unable to connect to server. Make sure the Flask app is running.');
            }
            
            throw error;
        }
    }

    /**
     * Test the server connection
     */
    async testConnection() {
        try {
            const result = await this.makeRequest('/test');
            console.log('✅ Server connection test passed');
            return result;
        } catch (error) {
            console.error('❌ Server connection test failed:', error.message);
            throw error;
        }
    }

    /**
     * Test the calculation endpoint with sample data
     */
    async testCalculation() {
        const sampleData = {
            users: 1000,
            churn: 0.15,
            ltv: 1200,
            monthly_expenses: 50000,
            initial_capital: 500000,
            cac: 150,
            market_segment: 'B2B',
            industry: 'Technology',
            prediction_horizon: 12,
            include_ml_predictions: true
        };

        try {
            const result = await this.makeRequest('/v2/calculate', sampleData, 'POST');
            console.log('✅ Calculation test passed');
            return result;
        } catch (error) {
            console.error('❌ Calculation test failed:', error.message);
            throw error;
        }
    }

    /**
     * Get health status
     */
    async getHealth() {
        try {
            const result = await this.makeRequest('/health');
            console.log('✅ Health check passed');
            return result;
        } catch (error) {
            console.error('❌ Health check failed:', error.message);
            throw error;
        }
    }

    /**
     * Calculate financial metrics
     */
    async calculateMetrics(data) {
        return await this.makeRequest('/v2/calculate', data, 'POST');
    }

    /**
     * Get AI advice
     */
    async getAdvice(prompt, context = {}) {
        const adviceData = {
            prompt: prompt,
            context: context,
            priority: 'NORMAL',
            language: 'en',
            max_response_length: 500
        };
        
        return await this.makeRequest('/v2/get-advice', adviceData, 'POST');
    }
}

// 🔥 USAGE EXAMPLES AND TESTING FUNCTIONS

// Initialize API client
const api = new FundPilotAPI('http://127.0.0.1:5000');

/**
 * Run comprehensive tests
 */
async function runAllTests() {
    console.log('🚀 Starting FundPilot API Tests');
    console.log('=' * 50);

    const tests = [
        { name: 'Server Connection', fn: () => api.testConnection() },
        { name: 'Health Check', fn: () => api.getHealth() },
        { name: 'Calculation Test', fn: () => api.testCalculation() }
    ];

    const results = [];

    for (const test of tests) {
        console.log(`\n🧪 Running test: ${test.name}`);
        try {
            const result = await test.fn();
            console.log(`✅ ${test.name}: PASSED`);
            results.push({ name: test.name, status: 'PASSED', result });
        } catch (error) {
            console.error(`❌ ${test.name}: FAILED - ${error.message}`);
            results.push({ name: test.name, status: 'FAILED', error: error.message });
        }
    }

    console.log('\n📊 Test Results Summary:');
    console.log('=' * 50);
    results.forEach(result => {
        const icon = result.status === 'PASSED' ? '✅' : '❌';
        console.log(`${icon} ${result.name}: ${result.status}`);
        if (result.error) {
            console.log(`   Error: ${result.error}`);
        }
    });

    return results;
}

/**
 * Simple calculation example
 */
async function calculateExample() {
    const data = {
        users: 500,
        churn: 0.1,
        ltv: 2000,
        monthly_expenses: 25000,
        initial_capital: 300000,
        cac: 200
    };

    try {
        console.log('🧮 Running calculation example...');
        const result = await api.calculateMetrics(data);
        console.log('✅ Calculation result:', result);
        return result;
    } catch (error) {
        console.error('❌ Calculation failed:', error.message);
        throw error;
    }
}

/**
 * Form handling example for HTML forms
 */
function setupFormHandler() {
    // Example form handler
    const form = document.getElementById('calculationForm');
    if (!form) {
        console.log('ℹ️ No form with ID "calculationForm" found');
        return;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        const data = {
            users: parseInt(formData.get('users')) || 0,
            churn: parseFloat(formData.get('churn')) || 0.1,
            ltv: parseFloat(formData.get('ltv')) || 0,
            monthly_expenses: parseFloat(formData.get('monthly_expenses')) || 0,
            initial_capital: parseFloat(formData.get('initial_capital')) || 0,
            cac: parseFloat(formData.get('cac')) || 0
        };

        console.log('📝 Form submitted with data:', data);

        try {
            // Show loading state
            const submitButton = form.querySelector('button[type="submit"]');
            const originalText = submitButton.textContent;
            submitButton.textContent = 'Calculating...';
            submitButton.disabled = true;

            const result = await api.calculateMetrics(data);
            
            // Display results
            displayResults(result);
            
        } catch (error) {
            displayError(error.message);
        } finally {
            // Reset button
            const submitButton = form.querySelector('button[type="submit"]');
            submitButton.textContent = originalText;
            submitButton.disabled = false;
        }
    });
}

/**
 * Display results in the UI
 */
function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    if (!resultsDiv) {
        console.log('ℹ️ No results div found, logging to console instead');
        console.log('📊 Results:', result);
        return;
    }

    if (result.success && result.data && result.data.summary_metrics) {
        const metrics = result.data.summary_metrics;
        resultsDiv.innerHTML = `
            <div class="results-success">
                <h3>✅ Calculation Results</h3>
                <div class="metrics-grid">
                    ${Object.entries(metrics).map(([key, value]) => `
                        <div class="metric-item">
                            <label>${key.replace(/_/g, ' ').toUpperCase()}</label>
                            <span>${typeof value === 'number' ? value.toLocaleString() : value}</span>
                        </div>
                    `).join('')}
                </div>
                <small>Request ID: ${result.request_id}</small>
            </div>
        `;
    } else {
        resultsDiv.innerHTML = `
            <div class="results-error">
                <h3>❌ Calculation Failed</h3>
                <p>Unexpected response format</p>
                <pre>${JSON.stringify(result, null, 2)}</pre>
            </div>
        `;
    }
}

/**
 * Display error in the UI
 */
function displayError(message) {
    const resultsDiv = document.getElementById('results');
    if (!resultsDiv) {
        console.error('❌ Error:', message);
        return;
    }

    resultsDiv.innerHTML = `
        <div class="results-error">
            <h3>❌ Error</h3>
            <p>${message}</p>
        </div>
    `;
}

// 🔥 AUTO-INITIALIZATION
document.addEventListener('DOMContentLoaded', () => {
    console.log('🔄 FundPilot API Client initialized');
    
    // Setup form handler if form exists
    setupFormHandler();
    
    // Add global functions for easy testing in console
    window.fundpilot = {
        api,
        runTests: runAllTests,
        calculate: calculateExample,
        testConnection: () => api.testConnection(),
        getHealth: () => api.getHealth()
    };
    
    console.log('🛠️ Available functions in window.fundpilot:');
    console.log('- runTests(): Run all API tests');
    console.log('- calculate(): Run sample calculation');
    console.log('- testConnection(): Test server connection');
    console.log('- getHealth(): Get server health status');
    console.log('- api: Direct API client access');
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FundPilotAPI, runAllTests, calculateExample };
}
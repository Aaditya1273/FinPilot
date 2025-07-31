document.addEventListener('DOMContentLoaded', () => {
    new FinPilot();
});

class FinPilot {
    constructor() {
        this.charts = new Map();
        this.apiKey = localStorage.getItem('finpilot_api_key') || null;
        this.modelData = null;

        this.config = {
            baseURL: 'http://127.0.0.1:5000/v2',
            retryAttempts: 3,
            retryDelay: 1000,
            debounceMs: 300,
        };

        this.elements = this.getElements();
        this.init();
    }

    init() {
        this.bindEvents();
        this.configureChartJs();
        this.checkApiKey();
    }

    getElements() {
        const elementIds = [
            'financial-form', 'generate-button', 'results-section', 'visuals-section',
            'ai-advisor-section', 'export-section', 'metrics-grid', 'mrr-metric',
            'capital-metric', 'ltv-cac-metric', 'burn-rate-metric', 'mrr-chart',
            'capital-chart', 'unit-economics-chart', 'ai-prompt', 'get-advice-button',
            'ai-response-content', 'export-button', 'api-key-modal', 'api-key-form',
            'api-key-input', 'modal-error', 'notification-container'
        ];
        return elementIds.reduce((acc, id) => {
            const camelCaseId = id.replace(/-(\w)/g, (_, c) => c.toUpperCase());
            acc[camelCaseId] = document.getElementById(id);
            return acc;
        }, {});
    }

    bindEvents() {
        this.elements.financialForm?.addEventListener('submit', this.handleGenerate.bind(this));
        this.elements.getAdviceButton?.addEventListener('click', this.handleGetAdvice.bind(this));
        this.elements.exportButton?.addEventListener('click', this.handleExport.bind(this));
        this.elements.apiKeyForm?.addEventListener('submit', this.handleApiKeySubmit.bind(this));
    }

    configureChartJs() {
        const style = getComputedStyle(document.body);
        Chart.defaults.color = style.getPropertyValue('--text-secondary').trim();
        Chart.defaults.borderColor = style.getPropertyValue('--border-color').trim();
        Chart.defaults.font.family = style.getPropertyValue('--font-sans');
    }

    checkApiKey() {
        if (!this.apiKey) {
            this.showModal(true);
        }
    }

    showModal(visible) {
        this.elements.apiKeyModal?.classList.toggle('hidden', !visible);
    }

    handleApiKeySubmit(event) {
        event.preventDefault();
        const newKey = this.elements.apiKeyInput.value.trim();
        if (newKey) {
            this.apiKey = newKey;
            localStorage.setItem('finpilot_api_key', newKey);
            this.elements.modalError.textContent = '';
            this.showModal(false);
            this.showNotification('API Key saved successfully!', 'success');
        } else {
            this.elements.modalError.textContent = 'Please enter a valid API key.';
        }
    }

    collectFormData() {
        const form = this.elements.financialForm;
        if (!form) return null;

        const data = {};
        const inputs = form.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            const key = input.id.replace(/-/g, '_');
            data[key] = parseFloat(input.value) || 0;
        });
        return data;
    }

    async handleGenerate(event) {
        event.preventDefault();
        if (!this.elements.financialForm.checkValidity()) {
            this.showNotification('Please fill in all required fields.', 'error');
            return;
        }

        this.setButtonLoading(this.elements.generateButton, true);

        try {
            const formData = this.collectFormData();
            const results = await this.apiCall('/calculate', { model_config: formData });
            this.modelData = results; // Store the full results
            this.updateUI(results);
            this.showSections(true);
            this.showNotification('Financial model generated successfully!', 'success');
        } catch (error) {
            this.handleApiError(error);
        } finally {
            this.setButtonLoading(this.elements.generateButton, false);
        }
    }

    async handleGetAdvice() {
        const prompt = this.elements.aiPrompt.value.trim();
        if (!prompt) {
            this.showNotification('Please enter a question for the AI advisor.', 'error');
            return;
        }

        this.setButtonLoading(this.elements.getAdviceButton, true);
        this.renderAIResponse(null, true); // Show skeleton loader

        try {
            const payload = { prompt, model_data: this.modelData };
            const result = await this.apiCall('/get-advice', payload);
            this.renderAIResponse(result.advice);
        } catch (error) {
            this.handleApiError(error);
            this.renderAIResponse(`**Error:** ${error.message}`);
        } finally {
            this.setButtonLoading(this.elements.getAdviceButton, false);
        }
    }

    async handleExport() {
        this.setButtonLoading(this.elements.exportButton, true);
        try {
            const canvas = await this.generateCombinedChart();
            const link = document.createElement('a');
            link.download = `finpilot-export-${Date.now()}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
        } catch (error) {
            this.showNotification(`Export failed: ${error.message}`, 'error');
        } finally {
            this.setButtonLoading(this.elements.exportButton, false);
        }
    }

    async apiCall(endpoint, payload) {
        if (!this.apiKey) {
            this.showModal(true);
            throw new Error('API Key is missing.');
        }

        let lastError;
        for (let i = 0; i < this.config.retryAttempts; i++) {
            try {
                const response = await fetch(`${this.config.baseURL}${endpoint}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-Key': this.apiKey,
                    },
                    body: JSON.stringify(payload),
                    signal: AbortSignal.timeout(15000) // 15-second timeout
                });

                if (response.ok) {
                    return await response.json();
                }

                const errorData = await response.json().catch(() => ({ error: `HTTP Error: ${response.status}` }));
                lastError = new Error(errorData.error || `HTTP Error: ${response.status}`);
                
                if (response.status === 401) { // Unauthorized
                    this.apiKey = null;
                    localStorage.removeItem('finpilot_api_key');
                    this.showModal(true);
                    this.elements.modalError.textContent = 'Your API key is invalid. Please enter a new one.';
                    throw lastError; // Stop retrying on auth error
                }

            } catch (error) {
                lastError = error;
                if (i < this.config.retryAttempts - 1) {
                    await new Promise(res => setTimeout(res, this.config.retryDelay * (i + 1)));
                } else {
                    throw lastError;
                }
            }
        }
    }

    handleApiError(error) {
        console.error('API Error:', error);
        this.showNotification(error.message || 'An unknown API error occurred.', 'error');
    }

    updateUI(data) {
        this.updateMetrics(data.summary_metrics);
        this.createOrUpdateCharts(data.projections);
    }

    updateMetrics(metrics) {
        const formatCurrency = (val) => `$${(val || 0).toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
        
        this.elements.mrrMetric.textContent = formatCurrency(metrics.final_mrr);
        this.elements.capitalMetric.textContent = formatCurrency(metrics.final_capital_remaining);
        this.elements.ltvCacMetric.textContent = (metrics.ltv_cac_ratio || 0).toFixed(2);
        this.elements.burnRateMetric.textContent = formatCurrency(metrics.average_burn_rate);

        this.elements.metricsGrid.classList.add('updated');
        setTimeout(() => this.elements.metricsGrid.classList.remove('updated'), 700);
    }

    createOrUpdateCharts(projections) {
        const labels = projections.month.map(m => `Month ${m}`);
        const chartConfigs = this.getChartConfigs(labels, projections);

        for (const [id, config] of Object.entries(chartConfigs)) {
            const chartInstance = this.charts.get(id);
            if (chartInstance) {
                chartInstance.data = config.data;
                chartInstance.update();
            } else {
                const ctx = this.elements[id]?.getContext('2d');
                if (ctx) {
                    this.charts.set(id, new Chart(ctx, config));
                }
            }
        }
    }

    getChartConfigs(labels, projections) {
        const style = getComputedStyle(document.body);
        const primaryColor = style.getPropertyValue('--primary').trim();
        const accentColor = style.getPropertyValue('--accent').trim();
        const dangerColor = style.getPropertyValue('--danger').trim();

        const getDataset = (label, data, color) => ({
            label,
            data,
            borderColor: color,
            backgroundColor: `${color}33`, // Add alpha transparency
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 5,
        });

        return {
            mrrChart: {
                type: 'line',
                data: { labels, datasets: [getDataset('MRR', projections.mrr, primaryColor)] },
                options: this.getChartOptions('Monthly Recurring Revenue')
            },
            capitalChart: {
                type: 'line',
                data: { labels, datasets: [getDataset('Capital Remaining', projections.capital_remaining, accentColor)] },
                options: this.getChartOptions('Capital Runway')
            },
            unitEconomicsChart: {
                type: 'bar',
                data: {
                    labels: ['LTV / CAC'],
                    datasets: [{
                        label: 'LTV/CAC Ratio',
                        data: [projections.ltv_cac_ratio.slice(-1)[0]],
                        backgroundColor: primaryColor,
                    }]
                },
                options: this.getChartOptions('LTV to CAC Ratio (Final Month)')
            }
        };
    }

    getChartOptions(titleText) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: { display: true, text: titleText, font: { size: 16 }, padding: { bottom: 20 } }
            },
            scales: {
                y: { beginAtZero: true, ticks: { callback: (val) => (val >= 1000 ? `${val / 1000}k` : val) } },
            },
            interaction: { intersect: false, mode: 'index' },
        };
    }

    renderAIResponse(markdown, isLoading = false) {
        if (isLoading) {
            this.elements.aiResponseContent.innerHTML = Array(3).fill('<div class="skeleton-loader"></div>').join('');
            return;
        }
        if (typeof marked !== 'undefined') {
            this.elements.aiResponseContent.innerHTML = marked.parse(markdown || '');
        } else {
            this.elements.aiResponseContent.textContent = markdown || '';
        }
    }

    showSections(visible) {
        const sections = [this.elements.resultsSection, this.elements.visualsSection, this.elements.aiAdvisorSection, this.elements.exportSection];
        sections.forEach(section => section?.classList.toggle('hidden', !visible));
    }

    setButtonLoading(button, isLoading) {
        if (!button) return;
        button.disabled = isLoading;
        const span = button.querySelector('span');
        const existingSpinner = button.querySelector('.loading-spinner');

        if (isLoading) {
            if (span) span.style.display = 'none';
            if (!existingSpinner) {
                const spinner = document.createElement('div');
                spinner.className = 'loading-spinner';
                button.prepend(spinner);
            }
        } else {
            if (span) span.style.display = 'inline';
            if (existingSpinner) existingSpinner.remove();
        }
    }

    showNotification(message, type = 'info') {
        const el = document.createElement('div');
        el.className = `notification notification-${type}`;
        el.textContent = message;
        this.elements.notificationContainer.appendChild(el);

        requestAnimationFrame(() => el.classList.add('show'));

        setTimeout(() => {
            el.classList.remove('show');
            el.addEventListener('transitionend', () => el.remove());
        }, 4000);
    }
    
    async generateCombinedChart() {
        const canvases = Array.from(this.charts.values()).map(chart => chart.canvas);
        if (canvases.length === 0) throw new Error('No charts available to export.');

        const padding = 20;
        const totalWidth = Math.max(...canvases.map(c => c.width));
        const totalHeight = canvases.reduce((sum, c) => sum + c.height, 0) + (padding * (canvases.length + 1));

        const canvas = document.createElement('canvas');
        canvas.width = totalWidth;
        canvas.height = totalHeight;
        canvas.height = 800;
        const ctx = canvas.getContext('2d');
        
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        return canvas;
    }

    cleanup() {
        this.destroyCharts();
        this.cache.clear();
        if (this.worker) {
            this.worker.terminate();
        }
        clearTimeout(this.debounceTimer);
    }
}

// Use the correct class name that was defined above
document.addEventListener('DOMContentLoaded', () => new FinPilot());
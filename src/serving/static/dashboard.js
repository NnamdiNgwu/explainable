  // Global variables
        let currentTab = 'overview';
        const API_BASE = '/api/v1';
        
        // Tab navigation
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.style.display = 'none';
            });
            
            // Show selected tab
            document.getElementById(tabName).style.display = 'block';
            
            // Update navigation
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });
            event.target.classList.add('active');
            
            currentTab = tabName;
            
            // Load tab-specific data
            loadTabData(tabName);
        }
        
         function showExplanationMethod(method) {
            document.querySelectorAll('.explanation-content').forEach(content => {
                content.style.display = 'none';
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            document.getElementById(method + 'Explanation').style.display = 'block';
            event.target.classList.add('active');
        }
        
        // Load tab-specific data
        async function loadTabData(tabName) {
            switch(tabName) {
                case 'overview':
                    await loadOverviewData();
                    break;
                case 'features':
                    await loadFeatureAnalysis();
                    break;
                case 'prediction':
                    setupPredictionForm();
                    break;
                case 'comparison':
                    // Model comparison is triggered manually
                    break;
                case 'explanations':
                    // Explanations are context-dependent
                    break;
            }
        }

        // Load overview data
        async function loadOverviewData() {
            try {
                const response = await fetch(`${API_BASE}/dashboard/summary`);
                const data = await response.json();
                
                // Update stats
                document.getElementById('totalPredictions').textContent = data.total_predictions.toLocaleString();
                document.getElementById('highRiskRate').textContent = (data.high_risk_rate * 100).toFixed(1) + '%';
                document.getElementById('avgConfidence').textContent = (data.average_confidence * 100).toFixed(1) + '%';
                document.getElementById('rfUsage').textContent = (data.model_distribution.rf_predictions * 100).toFixed(1) + '%';
                
                // Create model distribution chart
                const ctx = document.getElementById('modelDistributionChart').getContext('2d');
                new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Random Forest', 'Transformer'],
                        datasets: [{
                            data: [
                                data.model_distribution.rf_predictions * 100,
                                data.model_distribution.transformer_predictions * 100
                            ],
                            backgroundColor: ['#667eea', '#764ba2']
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'bottom'
                            }
                        }
                    }
                });

                // Update system health
                document.getElementById('systemHealth').innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>Status: <strong style="color: #2ecc71;">${data.system_health.status.toUpperCase()}</strong></span>
                        <span>Uptime: <strong>${data.system_health.uptime}</strong></span>
                        <span>Avg Response: <strong>${data.system_health.avg_response_time}</strong></span>
                    </div>
                `;
                
            } catch (error) {
                console.error('Failed to load overview data:', error);
            }
        }

        // Load feature analysis
        async function loadFeatureAnalysis() {
            try {
                const response = await fetch(`${API_BASE}/dashboard/feature-importance`);
                const data = await response.json();
                
                // Feature importance chart
                const ctx1 = document.getElementById('featureImportanceChart').getContext('2d');
                new Chart(ctx1, {
                    type: 'horizontalBar',
                    data: {
                        labels: data.rf_global_importance.slice(0, 10).map(f => f.feature),
                        datasets: [{
                            label: 'Importance',
                            data: data.rf_global_importance.slice(0, 10).map(f => f.importance),
                            backgroundColor: 'rgba(102, 126, 234, 0.7)',
                            borderColor: 'rgba(102, 126, 234, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            xAxes: [{
                                beginAtZero: true
                            }]
                        }
                    }
                });

                 // Feature categories chart
                const categories = Object.keys(data.feature_categories);
                const categoryImportances = categories.map(cat => data.feature_categories[cat].avg_importance);
                
                const ctx2 = document.getElementById('featureCategoriesChart').getContext('2d');
                new Chart(ctx2, {
                    type: 'bar',
                    data: {
                        labels: categories.map(cat => cat.charAt(0).toUpperCase() + cat.slice(1)),
                        datasets: [{
                            label: 'Average Importance',
                            data: categoryImportances,
                            backgroundColor: ['#667eea', '#764ba2', '#f093fb', '#f5576c']
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            yAxes: [{
                                beginAtZero: true
                            }]
                        }
                    }
                });
                
            } catch (error) {
                console.error('Failed to load feature analysis:', error);
            }
        }

        // Setup prediction form
        function setupPredictionForm() {
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const eventData = {};
                
                for (let [key, value] of formData.entries()) {
                    if (value === 'true' || value === 'false') {
                        eventData[key] = value === 'true';
                    } else if (!isNaN(value)) {
                        eventData[key] = parseFloat(value);
                    } else {
                        eventData[key] = value;
                    }
                }
                
                await makePrediction(eventData);
            });
        }

        // Make prediction
        async function makePrediction(eventData) {
            const resultsDiv = document.getElementById('predictionResults');
            resultsDiv.innerHTML = '<div class="loading">Making prediction...</div>';
            
            try {
                const response = await fetch(`${API_BASE}/predict`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(eventData)
                });
                
                const result = await response.json();

                 // Display results
                const riskClass = result.risk === 1 ? 'high' : 'low';
                const riskLabel = result.risk === 1 ? 'HIGH RISK' : 'LOW RISK';
                
                resultsDiv.innerHTML = `
                    <div style="margin-bottom: 20px;">
                        <span class="risk-badge risk-${riskClass}">${riskLabel}</span>
                        <div style="margin-top: 8px;">
                            <strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%
                        </div>
                        <div>
                            <strong>Model Used:</strong> ${result.explanation.model_used}
                        </div>
                    </div>
                    
                    <div>
                        <strong>Top Contributing Features:</strong>
                        <ul class="feature-list">
                            ${result.explanation.feature_ranks.slice(0, 5).map(feature => `
                                <li class="feature-item">
                                    <span>${feature}</span>
                                    <div class="feature-bar" style="width: ${Math.random() * 80 + 20}%"></div>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                `;

                // Generate SHAP waterfall plot
                await generateShapWaterfall(eventData);
                
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error">Prediction failed: ${error.message}</div>`;
            }
        }
        
        // Generate SHAP waterfall plot
        async function generateShapWaterfall(eventData) {
            try {
                const response = await fetch(`${API_BASE}/dashboard/shap-waterfall`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(eventData)
                });
                
                const data = await response.json();

                // Create Plotly waterfall chart
                const trace = {
                    type: 'waterfall',
                    orientation: 'v',
                    x: data.waterfall_data.map(d => d.feature),
                    y: data.waterfall_data.map(d => d.shap_value),
                    connector: {
                        line: {
                            color: 'rgb(63, 63, 63)'
                        }
                    },
                    decreasing: {
                        marker: {
                            color: 'rgb(244, 117, 96)'
                        }
                    },
                    increasing: {
                        marker: {
                            color: 'rgb(102, 126, 234)'
                        }
                    },
                    totals: {
                        marker: {
                            color: 'rgb(118, 75, 162)'
                        }
                    }
                };

                 const layout = {
                    title: 'SHAP Feature Contributions',
                    xaxis: {
                        type: 'category'
                    },
                    yaxis: {
                        title: 'SHAP Value'
                    },
                    showlegend: false
                };
                
                Plotly.newPlot('shapWaterfall', [trace], layout);
                
            } catch (error) {
                document.getElementById('shapWaterfall').innerHTML = `<div class="error">Failed to generate SHAP plot: ${error.message}</div>`;
            }
        }

        // Run model comparison
        async function runModelComparison() {
            const sampleEvent = {
                megabytes_sent: 15.7,
                hour: 22,
                destination_domain: "suspicious.net",
                channel: "HTTP",
                first_time_destination: true,
                after_hours: true
            };
            
            try {
                const response = await fetch(`${API_BASE}/dashboard/model-comparison`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(sampleEvent)
                });
                
                const data = await response.json();

                // Update RF results
                const rfDiv = document.getElementById('rfResults');
                const rfClass = data.rf_analysis.prediction === 1 ? 'high' : 'low';
                rfDiv.innerHTML = `
                    <div class="risk-badge risk-${rfClass}">
                        ${data.rf_analysis.prediction === 1 ? 'HIGH RISK' : 'LOW RISK'}
                    </div>
                    <div style="margin-top: 12px;">
                        <strong>Confidence:</strong> ${(data.rf_analysis.confidence * 100).toFixed(1)}%
                    </div>
                    <div style="margin-top: 8px;">
                        <strong>Top Features:</strong>
                        <ul style="margin-top: 8px; font-size: 12px;">
                            ${data.rf_analysis.top_features.slice(0, 3).map(([feature, importance]) => `
                                <li>${feature}: ${importance.toFixed(3)}</li>
                            `).join('')}
                        </ul>
                    </div>
                `;

                 // Update Transformer results
                const transformerDiv = document.getElementById('transformerResults');
                const transformerClass = data.transformer_analysis.prediction === 1 ? 'high' : 'low';
                transformerDiv.innerHTML = `
                    <div class="risk-badge risk-${transformerClass}">
                        ${data.transformer_analysis.prediction === 1 ? 'HIGH RISK' : 'LOW RISK'}
                    </div>
                    <div style="margin-top: 12px;">
                        <strong>Confidence:</strong> ${(data.transformer_analysis.confidence * 100).toFixed(1)}%
                    </div>
                    <div style="margin-top: 8px;">
                        <strong>Top Features:</strong>
                        <ul style="margin-top: 8px; font-size: 12px;">
                            ${data.transformer_analysis.top_features.slice(0, 3).map(([feature, importance]) => `
                                <li>${feature}: ${importance.toFixed(3)}</li>
                            `).join('')}
                        </ul>
                    </div>
                `;

                // Highlight winner
                if (data.cascade_decision.would_use_rf) {
                    document.querySelector('#rfResults').parentElement.classList.add('winner');
                    document.querySelector('#transformerResults').parentElement.classList.remove('winner');
                } else {
                    document.querySelector('#transformerResults').parentElement.classList.add('winner');
                    document.querySelector('#rfResults').parentElement.classList.remove('winner');
                }
                
            } catch (error) {
                console.error('Model comparison failed:', error);
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', () => {
            loadTabData('overview');
        });
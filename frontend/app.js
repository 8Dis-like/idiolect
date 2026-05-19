// Initialize Style Radar Chart
let radarChart;

function initChart() {
    const ctx = document.getElementById('styleRadar').getContext('2d');
    
    // Default empty data
    const data = {
        labels: ['Modularity', 'Docstrings', 'Typing', 'Naming', 'Complexity'],
        datasets: [{
            label: 'Base Model',
            data: [40, 30, 20, 50, 40],
            backgroundColor: 'rgba(148, 163, 184, 0.2)',
            borderColor: 'rgba(148, 163, 184, 0.8)',
            pointBackgroundColor: 'rgba(148, 163, 184, 1)',
        }, {
            label: 'Your Style (LoRA)',
            data: [0, 0, 0, 0, 0],
            backgroundColor: 'rgba(16, 185, 129, 0.2)',
            borderColor: 'rgba(16, 185, 129, 1)',
            pointBackgroundColor: 'rgba(16, 185, 129, 1)',
        }]
    };

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#94a3b8', font: { family: 'Inter', size: 10 } },
                    ticks: { display: false, min: 0, max: 100 }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#f8fafc', font: { family: 'Inter', size: 11 } }
                }
            }
        }
    });
}

// DOM Elements
const apiUrlInput = document.getElementById('api-url');
const promptInput = document.getElementById('prompt-input');
const generateBtn = document.getElementById('generate-btn');
const codeOutput = document.getElementById('code-output');
const statusIndicator = document.getElementById('status-indicator');

function setStatus(text, className) {
    statusIndicator.textContent = text;
    statusIndicator.className = `status-${className}`;
}

async function handleGenerate() {
    const prompt = promptInput.value.trim();
    const apiUrl = apiUrlInput.value.trim().replace(/\/$/, ""); // Remove trailing slash
    
    if (!prompt) return;

    // UI Loading State
    generateBtn.disabled = true;
    generateBtn.textContent = "Generating...";
    setStatus("Connecting to AWS...", "loading");
    codeOutput.textContent = "Processing prompt...";

    try {
        // 1. Generate Code
        const response = await fetch(`${apiUrl}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, max_new_tokens: 100 })
        });

        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
        
        const data = await response.json();
        const generatedCode = data.generated_text;
        
        codeOutput.textContent = generatedCode;
        setStatus("Analyzing style...", "loading");

        // 2. Analyze Style Fingerprint
        const styleResponse = await fetch(`${apiUrl}/analyze-style`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: generatedCode })
        });

        if (!styleResponse.ok) throw new Error("Style analysis failed");
        
        const styleData = await styleResponse.json();
        const metrics = styleData.metrics;

        // Map backend metrics to radar chart (0-100 scale simulation for demo)
        // In a real app, the backend would normalize these.
        const radarData = [
            Math.min(100, metrics.functions_count * 20 || 50), // Modularity
            Math.min(100, metrics.docstrings_count * 40 || 10), // Docstrings
            Math.min(100, metrics.type_hints_count * 30 || 20), // Typing
            Math.min(100, metrics.avg_name_length * 5 || 60),  // Naming
            Math.max(20, 100 - (metrics.max_indentation * 10)) // Complexity
        ];

        updateRadarChart(radarData);
        setStatus("Complete", "success");

    } catch (error) {
        console.error("API Error:", error);
        codeOutput.textContent = `Error: Could not connect to the API.\n\nMake sure your EC2 instance is running and the API URL is correct (${apiUrl}).\n\nDetails: ${error.message}`;
        setStatus("Failed", "error");
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = "Generate Code";
    }
}

function updateRadarChart(newData) {
    // Update the "Your Style" dataset
    radarChart.data.datasets[1].data = newData;
    radarChart.update();
}

// Event Listeners
document.addEventListener('DOMContentLoaded', initChart);
generateBtn.addEventListener('click', handleGenerate);

// Allow Ctrl+Enter to generate
promptInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        handleGenerate();
    }
});

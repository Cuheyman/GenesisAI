// Add to api-server/services/performanceMonitor.js
const prometheus = require('prom-client');

class PerformanceMonitor {
    constructor() {
        // Create metrics
        this.metrics = {
            apiLatency: new prometheus.Histogram({
                name: 'api_request_duration_seconds',
                help: 'API request latency',
                labelNames: ['method', 'route', 'status']
            }),
            
            tradingDecisions: new prometheus.Counter({
                name: 'trading_decisions_total',
                help: 'Total trading decisions made',
                labelNames: ['action', 'symbol', 'source']
            }),
            
            profitLoss: new prometheus.Gauge({
                name: 'portfolio_profit_loss',
                help: 'Current portfolio P&L',
                labelNames: ['type']
            }),
            
            modelAccuracy: new prometheus.Gauge({
                name: 'model_prediction_accuracy',
                help: 'Model prediction accuracy',
                labelNames: ['model', 'timeframe']
            }),
            
            systemHealth: new prometheus.Gauge({
                name: 'system_health_score',
                help: 'Overall system health score',
                labelNames: ['component']
            })
        };

        // Register metrics
        prometheus.register.registerMetric(this.metrics.apiLatency);
        prometheus.register.registerMetric(this.metrics.tradingDecisions);
        prometheus.register.registerMetric(this.metrics.profitLoss);
        prometheus.register.registerMetric(this.metrics.modelAccuracy);
        prometheus.register.registerMetric(this.metrics.systemHealth);
    }

    recordApiCall(method, route, status, duration) {
        this.metrics.apiLatency.observe(
            { method, route, status: status.toString() },
            duration
        );
    }

    recordTradingDecision(action, symbol, source) {
        this.metrics.tradingDecisions.inc({ action, symbol, source });
    }

    updateProfitLoss(realized, unrealized) {
        this.metrics.profitLoss.set({ type: 'realized' }, realized);
        this.metrics.profitLoss.set({ type: 'unrealized' }, unrealized);
    }

    updateModelAccuracy(model, timeframe, accuracy) {
        this.metrics.modelAccuracy.set({ model, timeframe }, accuracy);
    }

    async checkSystemHealth() {
        const health = {
            api: await this.checkApiHealth(),
            database: await this.checkDatabaseHealth(),
            ml_model: await this.checkModelHealth(),
            market_data: await this.checkMarketDataHealth()
        };

        // Update metrics
        Object.entries(health).forEach(([component, score]) => {
            this.metrics.systemHealth.set({ component }, score);
        });

        return health;
    }
}
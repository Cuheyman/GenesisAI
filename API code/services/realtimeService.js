// Add to api-server/services/realtimeService.js
const WebSocket = require('ws');
const EventEmitter = require('events');

class RealtimeDataService extends EventEmitter {
    constructor() {
        super();
        this.connections = new Map();
        this.dataStreams = new Map();
    }

    async connectToExchange(exchange, symbols) {
        const ws = new WebSocket(`wss://${exchange}.com/stream`);
        
        ws.on('message', (data) => {
            const parsed = JSON.parse(data);
            this.emit('market-update', {
                exchange,
                symbol: parsed.symbol,
                data: parsed
            });
        });

        this.connections.set(exchange, ws);
        
        // Subscribe to multiple data streams
        ws.send(JSON.stringify({
            method: 'SUBSCRIBE',
            params: [
                ...symbols.map(s => `${s}@trade`),
                ...symbols.map(s => `${s}@depth20`),
                ...symbols.map(s => `${s}@kline_1m`)
            ]
        }));
    }

    getAggregatedData(symbol) {
        // Combine data from all exchanges
        const aggregated = {
            symbol,
            bestBid: null,
            bestAsk: null,
            volume24h: 0,
            exchanges: []
        };

        for (const [exchange, data] of this.dataStreams) {
            if (data[symbol]) {
                // Aggregate best prices
                if (!aggregated.bestBid || data[symbol].bid > aggregated.bestBid) {
                    aggregated.bestBid = data[symbol].bid;
                }
                if (!aggregated.bestAsk || data[symbol].ask < aggregated.bestAsk) {
                    aggregated.bestAsk = data[symbol].ask;
                }
                aggregated.volume24h += data[symbol].volume || 0;
                aggregated.exchanges.push({ exchange, ...data[symbol] });
            }
        }

        return aggregated;
    }
}
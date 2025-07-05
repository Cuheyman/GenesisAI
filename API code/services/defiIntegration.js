// Add to api-server/services/defiIntegration.js
const { ethers } = require('ethers');
const { AlphaRouter } = require('@uniswap/smart-order-router');

class DeFiIntegration {
    constructor() {
        this.provider = new ethers.providers.JsonRpcProvider(process.env.ETH_RPC_URL);
        this.signer = new ethers.Wallet(process.env.PRIVATE_KEY, this.provider);
        this.router = new AlphaRouter({ chainId: 1, provider: this.provider });
    }

    async getOptimalDexRoute(tokenIn, tokenOut, amount) {
        // Compare prices across multiple DEXs
        const dexes = ['uniswap', 'sushiswap', '1inch', 'curve'];
        const quotes = await Promise.all(
            dexes.map(dex => this.getQuote(dex, tokenIn, tokenOut, amount))
        );

        // Find best route
        const bestQuote = quotes.reduce((best, current) => 
            current.outputAmount > best.outputAmount ? current : best
        );

        return {
            dex: bestQuote.dex,
            route: bestQuote.route,
            outputAmount: bestQuote.outputAmount,
            priceImpact: bestQuote.priceImpact,
            gasEstimate: bestQuote.gasEstimate
        };
    }

    async executeSmartTrade(params) {
        const { tokenIn, tokenOut, amount, slippage } = params;

        // Get optimal route
        const route = await this.getOptimalDexRoute(tokenIn, tokenOut, amount);

        // Build transaction with MEV protection
        const tx = await this.buildProtectedTransaction(route, slippage);

        // Use flashbots for private mempool
        const flashbotsProvider = await this.getFlashbotsProvider();
        const signedTx = await flashbotsProvider.signTransaction(tx);

        // Send transaction
        const receipt = await flashbotsProvider.sendRawTransaction(signedTx);

        return {
            txHash: receipt.hash,
            route: route.dex,
            executedPrice: receipt.effectivePrice,
            savedFromMev: this.calculateMevSavings(receipt)
        };
    }
}
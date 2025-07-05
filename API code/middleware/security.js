// Add to api-server/middleware/security.js
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');

class SecurityMiddleware {
    constructor() {
        this.jwtSecret = process.env.JWT_SECRET;
        this.apiKeys = new Map(); // Store in Redis in production
    }

    // Rate limiting with different tiers
    createRateLimiter(tier = 'standard') {
        const limits = {
            standard: { windowMs: 60000, max: 60 },
            premium: { windowMs: 60000, max: 300 },
            vip: { windowMs: 60000, max: 1000 }
        };

        return rateLimit({
            windowMs: limits[tier].windowMs,
            max: limits[tier].max,
            handler: (req, res) => {
                res.status(429).json({
                    error: 'Too many requests',
                    retryAfter: req.rateLimit.resetTime
                });
            }
        });
    }

    // Request signing verification
    verifyRequestSignature(req, res, next) {
        const signature = req.headers['x-signature'];
        const timestamp = req.headers['x-timestamp'];
        const nonce = req.headers['x-nonce'];

        if (!signature || !timestamp || !nonce) {
            return res.status(401).json({ error: 'Missing security headers' });
        }

        // Check timestamp (prevent replay attacks)
        const requestTime = parseInt(timestamp);
        const currentTime = Date.now();
        if (Math.abs(currentTime - requestTime) > 30000) { // 30 seconds
            return res.status(401).json({ error: 'Request expired' });
        }

        // Verify signature
        const payload = `${timestamp}${nonce}${JSON.stringify(req.body)}`;
        const expectedSignature = crypto
            .createHmac('sha256', req.apiSecret)
            .update(payload)
            .digest('hex');

        if (signature !== expectedSignature) {
            return res.status(401).json({ error: 'Invalid signature' });
        }

        next();
    }

    // IP whitelisting
    ipWhitelist(allowedIPs) {
        return (req, res, next) => {
            const clientIP = req.ip || req.connection.remoteAddress;
            
            if (!allowedIPs.includes(clientIP)) {
                return res.status(403).json({ error: 'IP not whitelisted' });
            }
            
            next();
        };
    }
}
"""
Cross-Exchange Arbitrage Engine - Opportunity Scanner
Profesjonalne skanowanie okazji arbitrażowych między giełdami
"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import numpy as np
import structlog
from abc import ABC, abstractmethod
from enum import Enum

logger = structlog.get_logger()

class ArbitrageType(Enum):
    """Typy arbitrażu"""
    SIMPLE = "simple"           # Prosta różnica cen między giełdami
    TRIANGULAR = "triangular"   # Arbitraż trójkątny na jednej giełdzie
    STATISTICAL = "statistical" # Arbitraż statystyczny (mean reversion)
    FUNDING = "funding"         # Arbitraż na funding rates

@dataclass
class ExchangePrice:
    """Cena na konkretnej giełdzie"""
    exchange: str
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime
    latency_ms: float = 0  # Opóźnienie API
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread_bps(self) -> float:
        if self.bid > 0:
            return ((self.ask - self.bid) / self.bid) * 10000
        return 10000  # Very high spread if invalid

@dataclass
class ArbitrageOpportunity:
    """Okazja arbitrażowa"""
    type: ArbitrageType
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_bps: float  # Zysk w basis points
    max_quantity: float  # Max ilość do arbitrażu
    confidence: float  # 0-1, pewność okazji
    estimated_profit_usd: float
    execution_time_ms: float  # Szacowany czas wykonania
    risk_score: float  # 0-100, im wyższy tym ryzykowniejszy
    created_at: datetime = field(default_factory=datetime.now)
    expiry_at: Optional[datetime] = None
    
    @property
    def is_expired(self) -> bool:
        if self.expiry_at is None:
            return False
        return datetime.now() > self.expiry_at
    
    @property
    def profit_percentage(self) -> float:
        return self.profit_bps / 10000

@dataclass
class ExchangeCapabilities:
    """Możliwości techniczne giełdy"""
    name: str
    api_latency_ms: float
    withdrawal_fees: Dict[str, float]  # Symbol -> fee
    min_trade_sizes: Dict[str, float]
    supports_margin: bool = False
    supports_futures: bool = False
    withdrawal_limits: Dict[str, float] = field(default_factory=dict)
    kyc_level_required: int = 1  # 1-3
    
class ArbitrageStrategy(ABC):
    """Bazowa klasa dla strategii arbitrażu"""
    
    @abstractmethod
    async def find_opportunities(
        self, 
        prices: Dict[str, List[ExchangePrice]],
        exchange_capabilities: Dict[str, ExchangeCapabilities]
    ) -> List[ArbitrageOpportunity]:
        pass

class SimpleArbitrageScanner(ArbitrageStrategy):
    """Skanuje proste różnice cen między giełdami"""
    
    def __init__(self, min_profit_bps: float = 20, min_volume_usd: float = 1000):
        self.min_profit_bps = min_profit_bps  # Min 0.2% profit
        self.min_volume_usd = min_volume_usd
        
    async def find_opportunities(
        self,
        prices: Dict[str, List[ExchangePrice]], 
        exchange_capabilities: Dict[str, ExchangeCapabilities]
    ) -> List[ArbitrageOpportunity]:
        """Znajdź okazje prostego arbitrażu"""
        
        opportunities = []
        
        for symbol, exchange_prices in prices.items():
            if len(exchange_prices) < 2:
                continue
                
            # Sort by ask price (cheapest first)
            sorted_by_ask = sorted(exchange_prices, key=lambda x: x.ask)
            # Sort by bid price (highest first)  
            sorted_by_bid = sorted(exchange_prices, key=lambda x: x.bid, reverse=True)
            
            best_buy = sorted_by_ask[0]   # Cheapest ask
            best_sell = sorted_by_bid[0]  # Highest bid
            
            # Check if different exchanges
            if best_buy.exchange == best_sell.exchange:
                continue
                
            # Calculate potential profit
            profit_per_unit = best_sell.bid - best_buy.ask
            if profit_per_unit <= 0:
                continue
                
            profit_bps = (profit_per_unit / best_buy.ask) * 10000
            
            if profit_bps < self.min_profit_bps:
                continue
                
            # Calculate max quantity considering liquidity
            max_buy_quantity = best_buy.ask_size
            max_sell_quantity = best_sell.bid_size
            max_quantity = min(max_buy_quantity, max_sell_quantity)
            
            # Check exchange capabilities
            buy_capabilities = exchange_capabilities.get(best_buy.exchange)
            sell_capabilities = exchange_capabilities.get(best_sell.exchange)
            
            if not buy_capabilities or not sell_capabilities:
                continue
                
            # Check minimum trade sizes
            min_trade_buy = buy_capabilities.min_trade_sizes.get(symbol, 0)
            min_trade_sell = sell_capabilities.min_trade_sizes.get(symbol, 0)
            max_quantity = max(0, max_quantity - max(min_trade_buy, min_trade_sell))
            
            if max_quantity <= 0:
                continue
                
            # Calculate profit after fees
            withdrawal_fee = buy_capabilities.withdrawal_fees.get(symbol, 0)
            net_profit_per_unit = profit_per_unit - withdrawal_fee
            
            if net_profit_per_unit <= 0:
                continue
                
            estimated_profit_usd = net_profit_per_unit * max_quantity
            
            if estimated_profit_usd < self.min_volume_usd:
                continue
                
            # Risk assessment
            risk_score = self._calculate_risk_score(
                best_buy, best_sell, buy_capabilities, sell_capabilities
            )
            
            # Confidence based on spreads and volumes
            confidence = self._calculate_confidence(
                best_buy, best_sell, profit_bps
            )
            
            # Execution time estimate
            execution_time_ms = (
                buy_capabilities.api_latency_ms + 
                sell_capabilities.api_latency_ms + 
                500  # Processing overhead
            )
            
            opportunity = ArbitrageOpportunity(
                type=ArbitrageType.SIMPLE,
                symbol=symbol,
                buy_exchange=best_buy.exchange,
                sell_exchange=best_sell.exchange, 
                buy_price=best_buy.ask,
                sell_price=best_sell.bid,
                profit_bps=profit_bps,
                max_quantity=max_quantity,
                confidence=confidence,
                estimated_profit_usd=estimated_profit_usd,
                execution_time_ms=execution_time_ms,
                risk_score=risk_score,
                expiry_at=datetime.now() + timedelta(seconds=30)  # 30s expiry
            )
            
            opportunities.append(opportunity)
            
            logger.debug(
                "Simple arbitrage opportunity found",
                symbol=symbol,
                buy_exchange=best_buy.exchange,
                sell_exchange=best_sell.exchange,
                profit_bps=profit_bps,
                profit_usd=estimated_profit_usd
            )
            
        return opportunities
    
    def _calculate_risk_score(
        self,
        buy_price: ExchangePrice,
        sell_price: ExchangePrice, 
        buy_capabilities: ExchangeCapabilities,
        sell_capabilities: ExchangeCapabilities
    ) -> float:
        """Oblicz score ryzyka (0-100)"""
        
        risk_score = 0
        
        # Spread risk - wide spreads increase risk
        avg_spread_bps = (buy_price.spread_bps + sell_price.spread_bps) / 2
        risk_score += min(30, avg_spread_bps / 5)  # Max 30 points
        
        # Latency risk 
        total_latency = buy_capabilities.api_latency_ms + sell_capabilities.api_latency_ms
        risk_score += min(20, total_latency / 50)  # Max 20 points
        
        # Volume risk - low volume increases risk
        min_volume = min(buy_price.ask_size, sell_price.bid_size)
        if min_volume < 1.0:
            risk_score += 25  # High risk for small volumes
        elif min_volume < 5.0:
            risk_score += 10
            
        # Exchange reliability (simplified)
        exchange_risk_map = {
            'binance': 5, 'coinbase': 5, 'kraken': 10,
            'bybit': 15, 'okx': 15, 'unknown': 25
        }
        
        buy_exchange_risk = exchange_risk_map.get(buy_capabilities.name.lower(), 25)
        sell_exchange_risk = exchange_risk_map.get(sell_capabilities.name.lower(), 25) 
        risk_score += (buy_exchange_risk + sell_exchange_risk) / 2  # Max ~25 points
        
        return min(100, risk_score)
    
    def _calculate_confidence(
        self,
        buy_price: ExchangePrice,
        sell_price: ExchangePrice,
        profit_bps: float
    ) -> float:
        """Oblicz confidence (0-1)"""
        
        confidence = 0.5  # Base confidence
        
        # Higher profit = higher confidence
        if profit_bps > 100:  # 1%+
            confidence += 0.3
        elif profit_bps > 50:  # 0.5%+
            confidence += 0.2
        elif profit_bps > 20:  # 0.2%+
            confidence += 0.1
            
        # Tighter spreads = higher confidence
        avg_spread_bps = (buy_price.spread_bps + sell_price.spread_bps) / 2
        if avg_spread_bps < 10:
            confidence += 0.15
        elif avg_spread_bps < 25:
            confidence += 0.05
            
        # Higher volumes = higher confidence
        min_volume = min(buy_price.ask_size, sell_price.bid_size)
        if min_volume > 10:
            confidence += 0.15
        elif min_volume > 5:
            confidence += 0.1
        elif min_volume > 1:
            confidence += 0.05
            
        # Recent price = higher confidence
        max_age_seconds = max(
            (datetime.now() - buy_price.timestamp).total_seconds(),
            (datetime.now() - sell_price.timestamp).total_seconds()
        )
        
        if max_age_seconds < 5:
            confidence += 0.1
        elif max_age_seconds < 30:
            confidence += 0.05
            
        return min(1.0, confidence)

class TriangularArbitrageScanner(ArbitrageStrategy):
    """Skanuje arbitraż trójkątny (np. BTC/USD -> ETH/BTC -> ETH/USD)"""
    
    def __init__(self, min_profit_bps: float = 15):
        self.min_profit_bps = min_profit_bps
        
    async def find_opportunities(
        self,
        prices: Dict[str, List[ExchangePrice]],
        exchange_capabilities: Dict[str, ExchangeCapabilities]
    ) -> List[ArbitrageOpportunity]:
        """Znajdź okazje arbitrażu trójkątnego"""
        
        opportunities = []
        
        # Group by exchange for triangular arbitrage
        exchange_prices = {}
        for symbol, price_list in prices.items():
            for price in price_list:
                if price.exchange not in exchange_prices:
                    exchange_prices[price.exchange] = {}
                exchange_prices[price.exchange][symbol] = price
                
        # Look for triangular opportunities on each exchange
        for exchange_name, symbol_prices in exchange_prices.items():
            triangular_opps = await self._find_triangular_on_exchange(
                exchange_name, symbol_prices, exchange_capabilities.get(exchange_name)
            )
            opportunities.extend(triangular_opps)
            
        return opportunities
    
    async def _find_triangular_on_exchange(
        self,
        exchange_name: str,
        symbol_prices: Dict[str, ExchangePrice],
        capabilities: Optional[ExchangeCapabilities]
    ) -> List[ArbitrageOpportunity]:
        """Znajdź triangular arbitrage na jednej giełdzie"""
        
        if not capabilities:
            return []
            
        opportunities = []
        
        # Common triangular patterns for crypto
        triangular_patterns = [
            ('BTCUSDT', 'ETHBTC', 'ETHUSDT'),
            ('BTCUSDT', 'ADABTC', 'ADAUSDT'),
            ('BTCUSDT', 'SOLBTC', 'SOLUSDT'),
            ('ETHUSDT', 'ADAETH', 'ADAUSDT'),
            # Add more patterns as needed
        ]
        
        for base_pair, cross_pair, target_pair in triangular_patterns:
            if all(pair in symbol_prices for pair in [base_pair, cross_pair, target_pair]):
                
                base_price = symbol_prices[base_pair]
                cross_price = symbol_prices[cross_pair] 
                target_price = symbol_prices[target_pair]
                
                # Calculate both directions
                
                # Direction 1: USDT -> BTC -> ALT -> USDT
                opp1 = self._calculate_triangular_profit(
                    base_price, cross_price, target_price, 
                    direction="forward", exchange_name=exchange_name
                )
                
                # Direction 2: USDT -> ALT -> BTC -> USDT  
                opp2 = self._calculate_triangular_profit(
                    base_price, cross_price, target_price,
                    direction="reverse", exchange_name=exchange_name
                )
                
                for opp in [opp1, opp2]:
                    if opp and opp.profit_bps >= self.min_profit_bps:
                        opportunities.append(opp)
                        
        return opportunities
        
    def _calculate_triangular_profit(
        self,
        base_price: ExchangePrice,    # BTCUSDT
        cross_price: ExchangePrice,   # ETHBTC
        target_price: ExchangePrice,  # ETHUSDT
        direction: str,
        exchange_name: str
    ) -> Optional[ArbitrageOpportunity]:
        """Oblicz zysk z arbitrażu trójkątnego"""
        
        if direction == "forward":
            # USDT -> BTC -> ETH -> USDT
            # Step 1: Buy BTC with USDT (use ask)
            btc_per_usdt = 1 / base_price.ask
            
            # Step 2: Buy ETH with BTC (use ask) 
            eth_per_btc = cross_price.ask
            
            # Step 3: Sell ETH for USDT (use bid)
            usdt_per_eth = target_price.bid
            
            # Final USDT amount from 1 USDT
            final_usdt = btc_per_usdt * eth_per_btc * usdt_per_eth
            
        else:  # reverse
            # USDT -> ETH -> BTC -> USDT
            # Step 1: Buy ETH with USDT (use ask)
            eth_per_usdt = 1 / target_price.ask
            
            # Step 2: Sell ETH for BTC (use bid)
            btc_per_eth = 1 / cross_price.bid if cross_price.bid > 0 else 0
            
            # Step 3: Sell BTC for USDT (use bid)
            usdt_per_btc = base_price.bid
            
            # Final USDT amount from 1 USDT
            final_usdt = eth_per_usdt * btc_per_eth * usdt_per_btc
            
        if final_usdt <= 1:
            return None
            
        profit_percentage = (final_usdt - 1) * 100  # Convert to percentage
        profit_bps = profit_percentage * 100
        
        # Estimate max quantity based on smallest liquidity
        min_liquidity = min(
            base_price.ask_size if direction == "forward" else base_price.bid_size,
            cross_price.ask_size if direction == "forward" else cross_price.bid_size,
            target_price.bid_size if direction == "forward" else target_price.ask_size
        )
        
        # Convert to USDT terms
        if direction == "forward":
            max_quantity_usdt = min_liquidity * base_price.ask
        else:
            max_quantity_usdt = min_liquidity * target_price.ask
            
        estimated_profit_usd = (final_usdt - 1) * max_quantity_usdt
        
        return ArbitrageOpportunity(
            type=ArbitrageType.TRIANGULAR,
            symbol=f"{base_price.symbol}-{cross_price.symbol}-{target_price.symbol}",
            buy_exchange=exchange_name,
            sell_exchange=exchange_name,  # Same exchange
            buy_price=1.0,  # Starting with 1 USDT
            sell_price=final_usdt,
            profit_bps=profit_bps,
            max_quantity=max_quantity_usdt,
            confidence=0.7,  # Generally lower confidence due to execution complexity
            estimated_profit_usd=estimated_profit_usd,
            execution_time_ms=1500,  # 3 trades = more time
            risk_score=60,  # Higher risk due to multiple trades
            expiry_at=datetime.now() + timedelta(seconds=15)  # Shorter expiry
        )

class StatisticalArbitrageScanner(ArbitrageStrategy):
    """Skanuje arbitraż statystyczny oparty na mean reversion"""
    
    def __init__(
        self, 
        lookback_periods: int = 24,
        z_score_threshold: float = 2.0,
        min_correlation: float = 0.8
    ):
        self.lookback_periods = lookback_periods
        self.z_score_threshold = z_score_threshold
        self.min_correlation = min_correlation
        self.price_history = {}  # Store price history
        
    async def find_opportunities(
        self,
        prices: Dict[str, List[ExchangePrice]],
        exchange_capabilities: Dict[str, ExchangeCapabilities]
    ) -> List[ArbitrageOpportunity]:
        """Znajdź okazje arbitrażu statystycznego"""
        
        # Update price history
        self._update_price_history(prices)
        
        opportunities = []
        
        # Look for mean reversion opportunities between correlated pairs
        correlated_pairs = self._find_correlated_pairs()
        
        for pair1, pair2, correlation in correlated_pairs:
            if correlation < self.min_correlation:
                continue
                
            # Calculate price ratio and z-score
            current_ratio = self._get_current_price_ratio(pair1, pair2, prices)
            if current_ratio is None:
                continue
                
            historical_ratios = self._get_historical_ratios(pair1, pair2)
            if len(historical_ratios) < self.lookback_periods:
                continue
                
            mean_ratio = np.mean(historical_ratios)
            std_ratio = np.std(historical_ratios)
            
            if std_ratio == 0:
                continue
                
            z_score = (current_ratio - mean_ratio) / std_ratio
            
            if abs(z_score) >= self.z_score_threshold:
                # Mean reversion opportunity
                if z_score > 0:
                    # Ratio is high - sell pair1, buy pair2
                    action_pair1 = "SELL"
                    action_pair2 = "BUY"
                else:
                    # Ratio is low - buy pair1, sell pair2
                    action_pair1 = "BUY"
                    action_pair2 = "SELL"
                    
                # Find best exchanges for execution
                best_exchanges = self._find_best_stat_arb_execution(
                    pair1, pair2, action_pair1, action_pair2, prices
                )
                
                if best_exchanges:
                    opportunity = self._create_stat_arb_opportunity(
                        pair1, pair2, z_score, correlation,
                        best_exchanges, current_ratio, mean_ratio
                    )
                    opportunities.append(opportunity)
                    
        return opportunities
        
    def _update_price_history(self, prices: Dict[str, List[ExchangePrice]]):
        """Aktualizuj historię cen"""
        current_time = datetime.now()
        
        for symbol, exchange_prices in prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                
            # Add current prices (use best bid/ask across exchanges)
            best_bid = max([p.bid for p in exchange_prices])
            best_ask = min([p.ask for p in exchange_prices])
            mid_price = (best_bid + best_ask) / 2
            
            self.price_history[symbol].append({
                'timestamp': current_time,
                'price': mid_price
            })
            
            # Keep only recent history
            cutoff_time = current_time - timedelta(hours=self.lookback_periods)
            self.price_history[symbol] = [
                p for p in self.price_history[symbol] 
                if p['timestamp'] > cutoff_time
            ]
            
    def _find_correlated_pairs(self) -> List[Tuple[str, str, float]]:
        """Znajdź skorelowane pary"""
        correlations = []
        symbols = list(self.price_history.keys())
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                pair1, pair2 = symbols[i], symbols[j]
                
                correlation = self._calculate_correlation(pair1, pair2)
                if correlation is not None:
                    correlations.append((pair1, pair2, correlation))
                    
        # Sort by correlation (highest first)
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        return correlations
        
    def _calculate_correlation(self, pair1: str, pair2: str) -> Optional[float]:
        """Oblicz korelację między dwoma parami"""
        history1 = self.price_history.get(pair1, [])
        history2 = self.price_history.get(pair2, [])
        
        if len(history1) < 10 or len(history2) < 10:
            return None
            
        # Align timestamps and calculate correlation
        prices1, prices2 = [], []
        
        # Simple alignment - use all available data points
        min_len = min(len(history1), len(history2))
        for i in range(min_len):
            prices1.append(history1[-(i+1)]['price'])
            prices2.append(history2[-(i+1)]['price'])
            
        if len(prices1) < 5:
            return None
            
        correlation_matrix = np.corrcoef(prices1, prices2)
        return correlation_matrix[0, 1]
        
    def _get_current_price_ratio(
        self, 
        pair1: str, 
        pair2: str,
        prices: Dict[str, List[ExchangePrice]]
    ) -> Optional[float]:
        """Pobierz aktualny stosunek cen"""
        
        if pair1 not in prices or pair2 not in prices:
            return None
            
        # Use best mid prices
        price1_list = prices[pair1]
        price2_list = prices[pair2]
        
        if not price1_list or not price2_list:
            return None
            
        price1 = np.mean([p.mid_price for p in price1_list])
        price2 = np.mean([p.mid_price for p in price2_list])
        
        if price2 == 0:
            return None
            
        return price1 / price2
        
    def _get_historical_ratios(self, pair1: str, pair2: str) -> List[float]:
        """Pobierz historyczne stosunki cen"""
        history1 = self.price_history.get(pair1, [])
        history2 = self.price_history.get(pair2, [])
        
        ratios = []
        min_len = min(len(history1), len(history2))
        
        for i in range(min_len):
            p1 = history1[-(i+1)]['price']
            p2 = history2[-(i+1)]['price']
            if p2 != 0:
                ratios.append(p1 / p2)
                
        return ratios
        
    def _find_best_stat_arb_execution(
        self,
        pair1: str,
        pair2: str, 
        action1: str,
        action2: str,
        prices: Dict[str, List[ExchangePrice]]
    ) -> Optional[Dict[str, str]]:
        """Znajdź najlepsze giełdy do wykonania stat arb"""
        
        pair1_prices = prices.get(pair1, [])
        pair2_prices = prices.get(pair2, [])
        
        if not pair1_prices or not pair2_prices:
            return None
            
        # Find best price for each action
        if action1 == "BUY":
            best_pair1 = min(pair1_prices, key=lambda x: x.ask)
        else:
            best_pair1 = max(pair1_prices, key=lambda x: x.bid)
            
        if action2 == "BUY":
            best_pair2 = min(pair2_prices, key=lambda x: x.ask)
        else:
            best_pair2 = max(pair2_prices, key=lambda x: x.bid)
            
        return {
            pair1: best_pair1.exchange,
            pair2: best_pair2.exchange
        }
        
    def _create_stat_arb_opportunity(
        self,
        pair1: str,
        pair2: str,
        z_score: float,
        correlation: float,
        exchanges: Dict[str, str],
        current_ratio: float,
        mean_ratio: float
    ) -> ArbitrageOpportunity:
        """Stwórz opportunity dla stat arb"""
        
        # Estimate profit potential based on z-score
        expected_reversion = abs(current_ratio - mean_ratio) / current_ratio
        profit_bps = expected_reversion * 10000 * 0.5  # Conservative estimate
        
        # Confidence based on z-score and correlation
        confidence = min(0.9, (abs(z_score) / 4.0) * (correlation ** 2))
        
        # Risk score based on correlation and z-score
        risk_score = max(20, 100 - (correlation * 50) - min(30, abs(z_score) * 5))
        
        return ArbitrageOpportunity(
            type=ArbitrageType.STATISTICAL,
            symbol=f"{pair1}/{pair2}",
            buy_exchange=exchanges.get(pair1, "unknown"),
            sell_exchange=exchanges.get(pair2, "unknown"),
            buy_price=current_ratio,
            sell_price=mean_ratio,
            profit_bps=profit_bps,
            max_quantity=10000,  # To be calculated based on available capital
            confidence=confidence,
            estimated_profit_usd=profit_bps * 10,  # Rough estimate
            execution_time_ms=2000,  # Two trades
            risk_score=risk_score,
            expiry_at=datetime.now() + timedelta(minutes=30)  # Longer expiry
        )

class ArbitrageOpportunityScanner:
    """Główny scanner dla wszystkich typów arbitrażu"""
    
    def __init__(
        self,
        exchange_clients: Dict[str, Any],
        exchange_capabilities: Dict[str, ExchangeCapabilities]
    ):
        self.exchange_clients = exchange_clients
        self.exchange_capabilities = exchange_capabilities
        
        # Initialize scanners
        self.scanners = {
            ArbitrageType.SIMPLE: SimpleArbitrageScanner(),
            ArbitrageType.TRIANGULAR: TriangularArbitrageScanner(),
            ArbitrageType.STATISTICAL: StatisticalArbitrageScanner()
        }
        
        self.active_opportunities: List[ArbitrageOpportunity] = []
        
    async def scan_all_opportunities(
        self,
        symbols: List[str],
        enabled_strategies: Set[ArbitrageType] = None
    ) -> List[ArbitrageOpportunity]:
        """Skanuj wszystkie typy arbitrażu"""
        
        if enabled_strategies is None:
            enabled_strategies = set(self.scanners.keys())
            
        # Get prices from all exchanges
        all_prices = await self._collect_prices(symbols)
        
        if not all_prices:
            logger.warning("No price data collected")
            return []
            
        logger.info(
            "Starting arbitrage scan",
            symbols=len(symbols),
            exchanges=len(self.exchange_clients),
            strategies=list(enabled_strategies)
        )
        
        # Run all enabled scanners
        all_opportunities = []
        
        for arb_type, scanner in self.scanners.items():
            if arb_type not in enabled_strategies:
                continue
                
            try:
                opportunities = await scanner.find_opportunities(
                    all_prices,
                    self.exchange_capabilities
                )
                
                all_opportunities.extend(opportunities)
                
                logger.info(
                    f"{arb_type.value} arbitrage scan completed",
                    opportunities_found=len(opportunities)
                )
                
            except Exception as e:
                logger.error(
                    f"{arb_type.value} arbitrage scan failed",
                    error=str(e)
                )
                
        # Filter and rank opportunities
        filtered_opportunities = self._filter_opportunities(all_opportunities)
        
        # Update active opportunities
        self.active_opportunities = filtered_opportunities
        
        logger.info(
            "Arbitrage scan completed",
            total_opportunities=len(filtered_opportunities),
            simple=len([o for o in filtered_opportunities if o.type == ArbitrageType.SIMPLE]),
            triangular=len([o for o in filtered_opportunities if o.type == ArbitrageType.TRIANGULAR]), 
            statistical=len([o for o in filtered_opportunities if o.type == ArbitrageType.STATISTICAL])
        )
        
        return filtered_opportunities
        
    async def _collect_prices(self, symbols: List[str]) -> Dict[str, List[ExchangePrice]]:
        """Zbierz ceny ze wszystkich giełd"""
        
        prices = {symbol: [] for symbol in symbols}
        
        # Collect from all exchanges in parallel
        tasks = []
        for exchange_name, client in self.exchange_clients.items():
            for symbol in symbols:
                task = self._get_exchange_price(exchange_name, symbol, client)
                tasks.append(task)
                
        # Execute all requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, ExchangePrice):
                prices[result.symbol].append(result)
            elif isinstance(result, Exception):
                logger.debug("Price fetch failed", error=str(result))
                
        return prices
        
    async def _get_exchange_price(
        self,
        exchange_name: str,
        symbol: str,
        client: Any
    ) -> Optional[ExchangePrice]:
        """Pobierz cenę z jednej giełdy"""
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # This would be specific to your exchange client implementation
            orderbook = await client.get_orderbook(symbol)
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
                return None
                
            best_bid = orderbook['bids'][0]
            best_ask = orderbook['asks'][0]
            
            return ExchangePrice(
                exchange=exchange_name,
                symbol=symbol,
                bid=best_bid['price'],
                ask=best_ask['price'], 
                bid_size=best_bid['quantity'],
                ask_size=best_ask['quantity'],
                timestamp=datetime.now(),
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.debug(
                "Failed to get price",
                exchange=exchange_name,
                symbol=symbol,
                error=str(e)
            )
            return None
            
    def _filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filtruj i rankuj opportunities"""
        
        # Remove expired
        active_opportunities = [
            opp for opp in opportunities 
            if not opp.is_expired
        ]
        
        # Remove low confidence
        filtered_opportunities = [
            opp for opp in active_opportunities
            if opp.confidence >= 0.3
        ]
        
        # Sort by expected profit (descending)
        filtered_opportunities.sort(
            key=lambda x: x.estimated_profit_usd,
            reverse=True
        )
        
        # Take top N
        return filtered_opportunities[:50]
        
    def get_best_opportunities(self, limit: int = 10) -> List[ArbitrageOpportunity]:
        """Pobierz najlepsze okazje"""
        return self.active_opportunities[:limit]
        
    def get_opportunities_by_type(self, arb_type: ArbitrageType) -> List[ArbitrageOpportunity]:
        """Pobierz okazje według typu"""
        return [opp for opp in self.active_opportunities if opp.type == arb_type]

# Przykład użycia:
"""
# Initialize exchange capabilities
exchange_capabilities = {
    'binance': ExchangeCapabilities(
        name='binance',
        api_latency_ms=50,
        withdrawal_fees={'BTCUSDT': 0.0001, 'ETHUSDT': 0.001},
        min_trade_sizes={'BTCUSDT': 0.001, 'ETHUSDT': 0.01}
    ),
    'bybit': ExchangeCapabilities(
        name='bybit', 
        api_latency_ms=75,
        withdrawal_fees={'BTCUSDT': 0.0002, 'ETHUSDT': 0.0015},
        min_trade_sizes={'BTCUSDT': 0.001, 'ETHUSDT': 0.01}
    )
}

# Initialize scanner
scanner = ArbitrageOpportunityScanner(
    exchange_clients={'binance': binance_client, 'bybit': bybit_client},
    exchange_capabilities=exchange_capabilities
)

# Scan for opportunities
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
opportunities = await scanner.scan_all_opportunities(symbols)

# Display results
for opp in opportunities[:5]:  # Top 5
    print(f"
    {opp.type.value.upper()} ARBITRAGE:
    Symbol: {opp.symbol}
    Buy: {opp.buy_exchange} @ ${opp.buy_price:.2f}
    Sell: {opp.sell_exchange} @ ${opp.sell_price:.2f}  
    Profit: {opp.profit_bps:.1f} bps (${opp.estimated_profit_usd:.2f})
    Confidence: {opp.confidence:.1%}
    Risk: {opp.risk_score:.0f}/100
    ")
"""

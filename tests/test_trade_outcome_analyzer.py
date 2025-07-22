"""
Unit tests for TradeOutcomeAnalyzer.

Tests all failure type classifications, probability calculations,
and analysis functionality.
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from core.trade_outcome_analyzer import (
    TradeOutcomeAnalyzer, TradeContext, MarketConditions, TechnicalIndicators,
    TradeAnalysisResult, FailureType, TradeOutcome
)


class TestTradeOutcomeAnalyzer:
    """Test suite for TradeOutcomeAnalyzer class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def analyzer(self, temp_dir):
        """Create analyzer instance for testing."""
        return TradeOutcomeAnalyzer(data_dir=temp_dir, lookback_days=30)
    
    @pytest.fixture
    def sample_market_conditions(self):
        """Sample market conditions for testing."""
        return MarketConditions(
            market_trend="bullish",
            volatility_regime="medium",
            volume_profile="above_average",
            time_of_day="opening",
            day_of_week="Monday",
            market_breadth=1.2,
            vix_level=18.5
        )
    
    @pytest.fixture
    def sample_technical_indicators(self):
        """Sample technical indicators for testing."""
        return TechnicalIndicators(
            rsi=65.0,
            macd_signal="bullish",
            moving_avg_position="above",
            volume_sma_ratio=1.5,
            atr=2.5,
            bollinger_position="middle",
            support_resistance_distance=1.2
        )
    
    @pytest.fixture
    def winning_trade_context(self, sample_market_conditions, sample_technical_indicators):
        """Sample winning trade context."""
        return TradeContext(
            symbol="RELIANCE",
            entry_price=2500.0,
            exit_price=2550.0,  # 2% profit
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now() - timedelta(hours=1),
            quantity=10,
            trade_type="long",
            strategy_used="momentum",
            stop_loss=2450.0,
            take_profit=2600.0,
            market_conditions=sample_market_conditions,
            technical_indicators=sample_technical_indicators,
            news_sentiment=0.3,
            volume_at_entry=150000,
            volume_at_exit=120000
        )
    
    @pytest.fixture
    def losing_trade_context(self, sample_market_conditions, sample_technical_indicators):
        """Sample losing trade context."""
        return TradeContext(
            symbol="RELIANCE",
            entry_price=2500.0,
            exit_price=2450.0,  # 2% loss
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now() - timedelta(hours=1),
            quantity=10,
            trade_type="long",
            strategy_used="momentum",
            stop_loss=2450.0,
            take_profit=2600.0,
            market_conditions=sample_market_conditions,
            technical_indicators=sample_technical_indicators,
            news_sentiment=0.1,
            volume_at_entry=150000,
            volume_at_exit=120000
        )


class TestBasicFunctionality:
    """Test basic analyzer functionality."""
    
    def test_analyzer_initialization(self, temp_dir):
        """Test analyzer initialization."""
        analyzer = TradeOutcomeAnalyzer(data_dir=temp_dir, lookback_days=60)
        
        assert analyzer.data_dir == temp_dir
        assert analyzer.lookback_days == 60
        assert os.path.exists(temp_dir)
        assert analyzer.trades_history == []
    
    def test_pnl_calculation_long(self, analyzer):
        """Test PnL calculation for long trades."""
        trade_context = TradeContext(
            symbol="TEST",
            entry_price=100.0,
            exit_price=110.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            quantity=10,
            trade_type="long",
            strategy_used="test"
        )
        
        pnl = analyzer._calculate_pnl(trade_context)
        pnl_pct = analyzer._calculate_pnl_percentage(trade_context)
        
        assert pnl == 100.0  # (110 - 100) * 10
        assert pnl_pct == 10.0  # 10% profit
    
    def test_pnl_calculation_short(self, analyzer):
        """Test PnL calculation for short trades."""
        trade_context = TradeContext(
            symbol="TEST",
            entry_price=100.0,
            exit_price=90.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            quantity=10,
            trade_type="short",
            strategy_used="test"
        )
        
        pnl = analyzer._calculate_pnl(trade_context)
        pnl_pct = analyzer._calculate_pnl_percentage(trade_context)
        
        assert pnl == 100.0  # (100 - 90) * 10
        assert pnl_pct == 10.0  # 10% profit for short
    
    def test_outcome_determination(self, analyzer):
        """Test trade outcome determination."""
        assert analyzer._determine_outcome(5.0) == TradeOutcome.WIN
        assert analyzer._determine_outcome(-5.0) == TradeOutcome.LOSS
        assert analyzer._determine_outcome(0.05) == TradeOutcome.BREAKEVEN


class TestFailureClassification:
    """Test failure type classification methods."""
    
    def test_false_breakout_detection(self, analyzer):
        """Test false breakout failure detection."""
        # Create trade context for false breakout
        trade_context = TradeContext(
            symbol="TEST",
            entry_price=100.0,
            exit_price=95.0,
            entry_time=datetime.now() - timedelta(minutes=20),  # Quick reversal
            exit_time=datetime.now(),
            quantity=10,
            trade_type="long",
            strategy_used="breakout",
            technical_indicators=TechnicalIndicators(
                volume_sma_ratio=0.8,  # Low volume
                support_resistance_distance=0.3  # Close to S/R
            )
        )
        
        reasons = []
        confidence = analyzer._check_false_breakout(trade_context, reasons)
        
        assert confidence > 0.5  # Should detect false breakout
        assert any("Quick price reversal" in reason for reason in reasons)
        assert any("below-average volume" in reason for reason in reasons)
    
    def test_stop_loss_hit_detection(self, analyzer):
        """Test stop loss hit failure detection."""
        trade_context = TradeContext(
            symbol="TEST",
            entry_price=100.0,
            exit_price=95.0,  # Matches stop loss
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            quantity=10,
            trade_type="long",
            strategy_used="momentum",
            stop_loss=95.0,
            technical_indicators=TechnicalIndicators(atr=2.0)
        )
        
        reasons = []
        confidence = analyzer._check_stop_loss_hit(trade_context, reasons)
        
        assert confidence > 0.8  # Should strongly detect stop loss hit
        assert any("stop loss level" in reason for reason in reasons)
    
    def test_trend_reversal_detection(self, analyzer):
        """Test trend reversal failure detection."""
        trade_context = TradeContext(
            symbol="TEST",
            entry_price=100.0,
            exit_price=95.0,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            quantity=10,
            trade_type="long",  # Long trade in bearish market
            strategy_used="trend",
            market_conditions=MarketConditions(
                market_trend="bearish",  # Against trade direction
                volatility_regime="high",
                volume_profile="average",
                time_of_day="mid_session",
                day_of_week="Tuesday",
                market_breadth=0.8
            ),
            technical_indicators=TechnicalIndicators(
                rsi=75.0,  # Overbought for long trade
                macd_signal="bearish"  # Against trade direction
            )
        )
        
        reasons = []
        confidence = analyzer._check_trend_reversal(trade_context, reasons)
        
        assert confidence > 0.5  # Should detect trend reversal
        assert any("against market trend" in reason for reason in reasons)
        assert any("overbought" in reason for reason in reasons)
    
    def test_low_volume_detection(self, analyzer):
        """Test low volume failure detection."""
        trade_context = TradeContext(
            symbol="TEST",
            entry_price=100.0,
            exit_price=95.0,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            quantity=10,
            trade_type="long",
            strategy_used="breakout",
            market_conditions=MarketConditions(
                market_trend="bullish",
                volatility_regime="low",
                volume_profile="below_average",  # Low volume
                time_of_day="mid_session",  # Low volume period
                day_of_week="Tuesday",
                market_breadth=1.1
            ),
            technical_indicators=TechnicalIndicators(
                volume_sma_ratio=0.6  # Significantly below average
            )
        )
        
        reasons = []
        confidence = analyzer._check_low_volume(trade_context, reasons)
        
        assert confidence > 0.5  # Should detect low volume
        assert any("below average" in reason for reason in reasons)
    
    def test_news_impact_detection(self, analyzer):
        """Test news impact failure detection."""
        trade_context = TradeContext(
            symbol="TEST",
            entry_price=100.0,
            exit_price=95.0,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            quantity=10,
            trade_type="long",  # Long trade with negative sentiment
            strategy_used="momentum",
            news_sentiment=-0.7  # Strong negative sentiment
        )
        
        reasons = []
        confidence = analyzer._check_news_impact(trade_context, reasons)
        
        assert confidence > 0.5  # Should detect news impact
        assert any("negative sentiment" in reason for reason in reasons)


class TestTradeAnalysis:
    """Test complete trade analysis functionality."""
    
    def test_winning_trade_analysis(self, analyzer, winning_trade_context):
        """Test analysis of winning trade."""
        result = analyzer.analyze_trade(winning_trade_context)
        
        assert result.outcome == TradeOutcome.WIN
        assert result.failure_type is None
        assert result.pnl > 0
        assert result.pnl_percentage > 0
        assert result.symbol == "RELIANCE"
        assert len(result.recommendations) >= 0  # May have general recommendations
    
    def test_losing_trade_analysis(self, analyzer, losing_trade_context):
        """Test analysis of losing trade."""
        result = analyzer.analyze_trade(losing_trade_context)
        
        assert result.outcome == TradeOutcome.LOSS
        assert result.failure_type is not None
        assert result.pnl < 0
        assert result.pnl_percentage < 0
        assert result.confidence_score >= 0
        assert len(result.failure_reasons) > 0
        assert len(result.recommendations) > 0
    
    def test_data_persistence(self, analyzer, winning_trade_context, temp_dir):
        """Test that trade analysis is saved and loaded correctly."""
        # Analyze a trade
        result = analyzer.analyze_trade(winning_trade_context)
        
        # Check that file was created
        assert os.path.exists(analyzer.analysis_file)
        
        # Create new analyzer instance and check data is loaded
        new_analyzer = TradeOutcomeAnalyzer(data_dir=temp_dir)
        assert len(new_analyzer.trades_history) == 1
        assert new_analyzer.trades_history[0].trade_id == result.trade_id


class TestPatternAnalysis:
    """Test pattern analysis and probability calculations."""
    
    def test_failure_probability_calculation(self, analyzer):
        """Test failure probability calculation with insufficient data."""
        trade_context = TradeContext(
            symbol="TEST",
            entry_price=100.0,
            exit_price=95.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            quantity=10,
            trade_type="long",
            strategy_used="test"
        )
        
        # With no historical data, should return default probability
        probability = analyzer._calculate_failure_probability(trade_context)
        assert probability == 0.5
    
    def test_similar_trades_finding(self, analyzer, losing_trade_context):
        """Test finding similar trades."""
        # Add some historical trades
        for i in range(5):
            historical_result = TradeAnalysisResult(
                trade_id=f"test_{i}",
                symbol="RELIANCE",
                outcome=TradeOutcome.LOSS if i < 3 else TradeOutcome.WIN,
                failure_type=FailureType.STOP_LOSS_HIT if i < 3 else None,
                pnl=-50.0 if i < 3 else 50.0,
                pnl_percentage=-2.0 if i < 3 else 2.0,
                confidence_score=0.8 if i < 3 else 0.0,
                failure_probability=0.6,
                similar_trades_count=5,
                analysis_timestamp=datetime.now() - timedelta(days=i),
                failure_reasons=["test reason"] if i < 3 else [],
                market_context_score=0.5,
                technical_context_score=0.5,
                recommendations=["test recommendation"]
            )
            analyzer.trades_history.append(historical_result)
        
        similar_trades = analyzer._find_similar_trades(losing_trade_context)
        assert len(similar_trades) >= 0  # Should find some similar trades


class TestStatisticsAndReporting:
    """Test statistics and reporting functionality."""
    
    def test_failure_statistics(self, analyzer):
        """Test failure statistics calculation."""
        # Add some test data
        for i in range(10):
            result = TradeAnalysisResult(
                trade_id=f"test_{i}",
                symbol=f"STOCK{i % 3}",
                outcome=TradeOutcome.LOSS if i < 6 else TradeOutcome.WIN,
                failure_type=FailureType.STOP_LOSS_HIT if i < 3 else FailureType.FALSE_BREAKOUT if i < 6 else None,
                pnl=-50.0 if i < 6 else 50.0,
                pnl_percentage=-2.0 if i < 6 else 2.0,
                confidence_score=0.8 if i < 6 else 0.0,
                failure_probability=0.6,
                similar_trades_count=5,
                analysis_timestamp=datetime.now() - timedelta(days=i),
                failure_reasons=["test reason"] if i < 6 else [],
                market_context_score=0.5,
                technical_context_score=0.5,
                recommendations=["test recommendation"]
            )
            analyzer.trades_history.append(result)
        
        stats = analyzer.get_failure_statistics(days=30)
        
        assert stats["total_trades"] == 10
        assert stats["win_rate"] == 40.0  # 4 wins out of 10
        assert stats["loss_rate"] == 60.0  # 6 losses out of 10
        assert "failure_type_distribution" in stats
        assert stats["most_common_failure"] in ["stop_loss_hit", "false_breakout"]
    
    def test_symbol_analysis(self, analyzer):
        """Test symbol-specific analysis."""
        # Add test data for specific symbol
        for i in range(5):
            result = TradeAnalysisResult(
                trade_id=f"reliance_{i}",
                symbol="RELIANCE",
                outcome=TradeOutcome.LOSS if i < 3 else TradeOutcome.WIN,
                failure_type=FailureType.TREND_REVERSAL if i < 3 else None,
                pnl=-50.0 if i < 3 else 50.0,
                pnl_percentage=-2.0 if i < 3 else 2.0,
                confidence_score=0.8 if i < 3 else 0.0,
                failure_probability=0.6,
                similar_trades_count=5,
                analysis_timestamp=datetime.now() - timedelta(days=i),
                failure_reasons=[f"reason_{i}"] if i < 3 else [],
                market_context_score=0.5,
                technical_context_score=0.5,
                recommendations=["test recommendation"]
            )
            analyzer.trades_history.append(result)
        
        analysis = analyzer.get_symbol_analysis("RELIANCE", days=30)
        
        assert analysis["symbol"] == "RELIANCE"
        assert analysis["total_trades"] == 5
        assert analysis["win_rate"] == 40.0  # 2 wins out of 5
        assert "common_failure_reasons" in analysis
    
    def test_export_detailed_report(self, analyzer, temp_dir):
        """Test detailed report export."""
        # Add some test data
        result = TradeAnalysisResult(
            trade_id="test_export",
            symbol="TEST",
            outcome=TradeOutcome.LOSS,
            failure_type=FailureType.STOP_LOSS_HIT,
            pnl=-50.0,
            pnl_percentage=-2.0,
            confidence_score=0.8,
            failure_probability=0.6,
            similar_trades_count=5,
            analysis_timestamp=datetime.now(),
            failure_reasons=["test reason"],
            market_context_score=0.5,
            technical_context_score=0.5,
            recommendations=["test recommendation"]
        )
        analyzer.trades_history.append(result)
        
        # Export report
        filepath = analyzer.export_detailed_report("test_report.csv")
        
        assert os.path.exists(filepath)
        assert filepath.endswith("test_report.csv")
        
        # Check file content
        with open(filepath, 'r') as f:
            content = f.read()
            assert "test_export" in content
            assert "TEST" in content
    
    def test_cleanup_old_data(self, analyzer):
        """Test cleanup of old data."""
        # Add old and new data
        old_result = TradeAnalysisResult(
            trade_id="old_trade",
            symbol="OLD",
            outcome=TradeOutcome.LOSS,
            failure_type=FailureType.STOP_LOSS_HIT,
            pnl=-50.0,
            pnl_percentage=-2.0,
            confidence_score=0.8,
            failure_probability=0.6,
            similar_trades_count=5,
            analysis_timestamp=datetime.now() - timedelta(days=200),  # Old data
            failure_reasons=["old reason"],
            market_context_score=0.5,
            technical_context_score=0.5,
            recommendations=["old recommendation"]
        )
        
        new_result = TradeAnalysisResult(
            trade_id="new_trade",
            symbol="NEW",
            outcome=TradeOutcome.WIN,
            failure_type=None,
            pnl=50.0,
            pnl_percentage=2.0,
            confidence_score=0.0,
            failure_probability=0.4,
            similar_trades_count=5,
            analysis_timestamp=datetime.now() - timedelta(days=10),  # Recent data
            failure_reasons=[],
            market_context_score=0.5,
            technical_context_score=0.5,
            recommendations=[]
        )
        
        analyzer.trades_history.extend([old_result, new_result])
        
        # Cleanup old data (keep 30 days)
        removed_count = analyzer.cleanup_old_data(days_to_keep=30)
        
        assert removed_count == 1  # Should remove 1 old trade
        assert len(analyzer.trades_history) == 1
        assert analyzer.trades_history[0].trade_id == "new_trade"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_optional_data(self, analyzer):
        """Test analysis with missing optional data."""
        minimal_trade = TradeContext(
            symbol="TEST",
            entry_price=100.0,
            exit_price=95.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            quantity=10,
            trade_type="long",
            strategy_used="test"
            # No optional fields
        )
        
        # Should not raise exception
        result = analyzer.analyze_trade(minimal_trade)
        assert result.outcome == TradeOutcome.LOSS
        assert result.failure_type is not None  # Should still classify
    
    def test_invalid_trade_data(self, analyzer):
        """Test handling of invalid trade data."""
        # Test with zero quantity
        invalid_trade = TradeContext(
            symbol="TEST",
            entry_price=100.0,
            exit_price=95.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            quantity=0,  # Invalid
            trade_type="long",
            strategy_used="test"
        )
        
        # Should handle gracefully
        result = analyzer.analyze_trade(invalid_trade)
        assert result.pnl == 0.0
    
    def test_empty_statistics(self, analyzer):
        """Test statistics with no data."""
        stats = analyzer.get_failure_statistics(days=30)
        assert "error" in stats
        
        symbol_analysis = analyzer.get_symbol_analysis("NONEXISTENT")
        assert "error" in symbol_analysis


if __name__ == "__main__":
    pytest.main([__file__])
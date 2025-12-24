from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
import numpy as np
from structure.failures.config import StructureBreakConfig
from structure.failures.entity import ResultBuilder, BreakLevel, LevelState, BreakTarget
from structure.failures.retests import RETEST_HANDLERS



class BarProcessor:
    """
    Processes individual bars for structure break detection.

    This class follows the same pattern as the original _process_bar:
    - Handles gap detection, breakout registration, and level updates
    - Leaves momentum continuation to the main loop (as in original)
    - Maintains compatibility with existing retest handlers
    """

    @staticmethod
    def _get_break_targets(
            swings: Dict[str, List[Dict[str, Any]]],
            trend: str
    ) -> Dict[str, Optional[BreakTarget]]:
        """
        Extract break targets from swing points based on current trend.

        :param swings: Dictionary of swing points
        :param trend: Current trend ('uptrend' or 'downtrend')
        :return: Dictionary of break targets by role
        """
        targets = {}

        # Break of Structure (BOS) targets
        if trend == 'uptrend' and swings.get('hh'):
            last_hh = swings['hh'][-1]
            targets['bos_bull'] = BreakTarget(
                role='bos_bull',
                direction='bullish',
                price=last_hh['price'],
                idx=last_hh['idx']
            )
        else:
            targets['bos_bull'] = None

        if trend == 'downtrend' and swings.get('ll'):
            last_ll = swings['ll'][-1]
            targets['bos_bear'] = BreakTarget(
                role='bos_bear',
                direction='bearish',
                price=last_ll['price'],
                idx=last_ll['idx']
            )
        else:
            targets['bos_bear'] = None

        # Change of Character (CHOCH) targets
        if trend == 'uptrend' and swings.get('hl'):
            last_hl = swings['hl'][-1]
            targets['choch_bear'] = BreakTarget(
                role='choch_bear',
                direction='bearish',
                price=last_hl['price'],
                idx=last_hl['idx']
            )
        else:
            targets['choch_bear'] = None

        if trend == 'downtrend' and swings.get('lh'):
            last_lh = swings['lh'][-1]
            targets['choch_bull'] = BreakTarget(
                role='choch_bull',
                direction='bullish',
                price=last_lh['price'],
                idx=last_lh['idx']
            )
        else:
            targets['choch_bull'] = None

        return targets

    @staticmethod
    def _detect_gap_breaks(
            targets: Dict[str, Optional[BreakTarget]],
            open_price: float,
            min_move: float,
            bar_index: int
    ) -> Dict[str, bool]:
        """
        Detect gap breaks at bar open.

        :param targets: Break targets dictionary
        :param open_price: Current bar's open price
        :param min_move: Minimum move required for a break
        :param bar_index: Current bar index
        :return: Dictionary indicating gap breaks by target role
        """
        if bar_index == 0:
            return {k: False for k in targets}

        is_gap = {k: False for k in targets}

        for role, target in targets.items():
            if target is None:
                continue

            if role == 'bos_bull' and open_price > target.price + min_move:
                is_gap[role] = True
            elif role == 'bos_bear' and open_price < target.price - min_move:
                is_gap[role] = True
            elif role == 'choch_bear' and open_price < target.price - min_move:
                is_gap[role] = True
            elif role == 'choch_bull' and open_price > target.price + min_move:
                is_gap[role] = True

        return is_gap

    @staticmethod
    def _check_breakout_conditions(
            target: BreakTarget,
            close: float,
            body_ratio: float,
            is_bullish_body: bool,
            is_bearish_body: bool,
            close_location: float,
            upper_wick_ratio: float,
            lower_wick_ratio: float,
            min_move: float,
            config: StructureBreakConfig
    ) -> bool:
        """
        Check if breakout conditions are met for a target.

        :param target: Break target to check
        :param close: Current close price
        :param body_ratio: Body ratio of current candle
        :param is_bullish_body: Whether candle is bullish
        :param is_bearish_body: Whether candle is bearish
        :param close_location: Close location within candle range
        :param upper_wick_ratio: Upper wick ratio
        :param lower_wick_ratio: Lower wick ratio
        :param min_move: Minimum move required
        :param config: Configuration object
        :return: True if breakout conditions are met
        """
        if target.direction == 'bullish':
            # Bullish breakout conditions
            base_cond = (
                    close > target.price + min_move and
                    body_ratio >= config.min_break_body_ratio and
                    is_bullish_body
            )
            quality_cond = (
                    close_location >= 0.7 and
                    upper_wick_ratio <= config.upper_wick_ratio
            )
            return base_cond and quality_cond
        else:
            # Bearish breakout conditions
            base_cond = (
                    close < target.price - min_move and
                    body_ratio >= config.min_break_body_ratio and
                    is_bearish_body
            )
            quality_cond = (
                    close_location <= 0.3 and
                    lower_wick_ratio <= config.upper_wick_ratio
            )
            return base_cond and quality_cond

    @staticmethod
    def _process_potential_breaks(
            targets: Dict[str, Optional[BreakTarget]],
            gap_breaks: Dict[str, bool],
            bar_index: int,
            close: float,
            high: float,
            low: float,
            body_ratio: float,
            is_bullish_body: bool,
            is_bearish_body: bool,
            close_location: float,
            upper_wick_ratio: float,
            lower_wick_ratio: float,
            atr_val: float,
            min_move: float,
            config: StructureBreakConfig,
            active_levels: Dict[Tuple[str, int], BreakLevel],
            builder: ResultBuilder
    ) -> None:
        """
        Process potential breakouts and create new break levels.

        :param targets: Break targets
        :param gap_breaks: Gap break indicators
        :param bar_index: Current bar index
        :param close: Close price
        :param high: High price
        :param low: Low price
        :param body_ratio: Body ratio
        :param is_bullish_body: Bullish flag
        :param is_bearish_body: Bearish flag
        :param close_location: Close location
        :param upper_wick_ratio: Upper wick ratio
        :param lower_wick_ratio: Lower wick ratio
        :param atr_val: ATR value
        :param min_move: Minimum move
        :param config: Configuration
        :param active_levels: Active break levels
        :param builder: Result builder
        """
        signal_map = {
            'bos_bull': 'is_bos_bullish_initial',
            'bos_bear': 'is_bos_bearish_initial',
            'choch_bear': 'is_choch_bearish',
            'choch_bull': 'is_choch_bullish'
        }

        for role, target in targets.items():
            if target is None:
                continue

            # Check breakout conditions
            should_break = BarProcessor._check_breakout_conditions(
                target=target,
                close=close,
                body_ratio=body_ratio,
                is_bullish_body=is_bullish_body,
                is_bearish_body=is_bearish_body,
                close_location=close_location,
                upper_wick_ratio=upper_wick_ratio,
                lower_wick_ratio=lower_wick_ratio,
                min_move=min_move,
                config=config
            )

            if should_break:
                key = (role, target.idx)
                if key not in active_levels:
                    level = BreakLevel(
                        swing_idx=target.idx,
                        price=target.price,
                        direction=target.direction,
                        role='bos' if 'bos' in role else 'choch',
                        break_idx=bar_index,
                        atr_at_break=atr_val,
                        is_gap_break=gap_breaks[role],
                        config=config
                    )

                    if not gap_breaks[role]:
                        if target.direction == 'bullish':
                            level.max_post_break_high = high
                        else:
                            level.min_post_break_low = low

                    active_levels[key] = level
                    builder.set_signal(signal_map[role], bar_index)


    @staticmethod
    def _update_active_level(
            level: BreakLevel,
            bar_index: int,
            close: float,
            high: float,
            low: float,
            prev_high: Optional[float],
            prev_low: Optional[float],
            config: StructureBreakConfig,
            builder: ResultBuilder
    ) -> Tuple[bool, Optional[str]]:
        """
        Update a single active break level.

        Note: Follows original pattern - momentum continuation is handled
        in the main loop (requires dataframe slice).

        :param level: Break level to update
        :param bar_index: Current bar index
        :param close: Close price
        :param high: High price
        :param low: Low price
        :param prev_high: Previous bar's high (for bullish retests)
        :param prev_low: Previous bar's low (for bearish retests)
        :param config: Configuration
        :param builder: Result builder
        :return: Tuple of (should_remove, failure_signal_name)

        Update a single active break level with follow-through validation.

        Note: Momentum continuation is still handled in the main loop.
        Retests and confirmation now require LevelState.CONFIRMED.
        """
        if bar_index < level.break_idx:
            return False, None

        bars_since_break = bar_index - level.break_idx

        # Update extremes and movement
        if level.direction == 'bullish':
            level.max_post_break_high = max(level.max_post_break_high, high)
            level.moved_away_distance = level.max_post_break_high - level.price
        else:
            level.min_post_break_low = min(level.min_post_break_low, low)
            level.moved_away_distance = level.price - level.min_post_break_low

        # === FOLLOW-THROUGH VALIDATION ===
        if level.state == LevelState.BROKEN:
            # Initialize follow-through start for gap breaks
            if level.follow_through_start is None:
                level.follow_through_start = bar_index

            ft_start = level.follow_through_start
            ft_window_end = ft_start + config.follow_through_bars

            if bar_index < ft_window_end:
                # Count qualifying closes
                qualifies = False
                if level.direction == 'bullish':
                    qualifies = close > level.price + level.buffer
                else:
                    qualifies = close < level.price - level.buffer
                if qualifies:
                    level.follow_through_progress += 1
            else:
                # Window ended — validate follow-through
                required = int(np.ceil(config.follow_through_close_ratio * config.follow_through_bars))
                if level.follow_through_progress >= required:
                    level.state = LevelState.CONFIRMED
                else:
                    level.state = LevelState.FAILED_IMMEDIATE
                    signal_name = (
                        'is_bos_bullish_failed_follow_through'
                        if level.direction == 'bullish' else
                        'is_bos_bearish_failed_follow_through'
                    )
                    return True, signal_name

        # === IMMEDIATE FAILURE (only during first 3 bars AND if still BROKEN) ===
        if (level.state == LevelState.BROKEN and
                bars_since_break <= 3):
            if level.direction == 'bullish' and close < level.price - level.buffer:
                signal_name = 'is_failed_choch_bearish' if level.role == 'choch' else 'is_bullish_immediate_failure'
                return True, signal_name
            elif level.direction == 'bearish' and close > level.price + level.buffer:
                signal_name = 'is_failed_choch_bullish' if level.role == 'choch' else 'is_bearish_immediate_failure'
                return True, signal_name

        # === RETEST PROCESSING (only on CONFIRMED levels) ===
        if (level.state == LevelState.CONFIRMED and
                config.pullback_min_bars <= bars_since_break <= config.pullback_max_bars):
            handler = RETEST_HANDLERS.get((level.role, level.direction))
            if handler:
                if level.direction == 'bullish':
                    handler(level, high, low, close, prev_high, bar_index, config, builder)
                else:
                    handler(level, high, low, close, prev_low, bar_index, config, builder)
                if level.state in (LevelState.CONFIRMED, LevelState.FAILED_RETEST):
                    return True, None

        return False, None

    @staticmethod
    def _cleanup_active_levels(
            active_levels: Dict[Tuple[str, int], BreakLevel],
            max_active_levels: int
    ) -> None:
        """
        Clean up old active levels to prevent memory bloat.

        :param active_levels: Active levels dictionary
        :param max_active_levels: Maximum number of active levels to keep
        """
        if len(active_levels) > max_active_levels:
            # Remove oldest level
            oldest_key = min(active_levels.keys(), key=lambda k: active_levels[k].break_idx)
            active_levels.pop(oldest_key, None)

    @staticmethod
    def process_bar(
            bar_index: int,
            close: float,
            high: float,
            low: float,
            open_price: float,
            atr_val: float,
            body_ratio: float,
            is_bullish_body: bool,
            is_bearish_body: bool,
            trend: str,
            prev_high: Optional[float],
            prev_low: Optional[float],
            swings: Dict[str, List[Dict[str, Any]]],
            active_levels: Dict[Tuple[str, int], BreakLevel],
            builder: ResultBuilder,
            config: StructureBreakConfig,
            close_location: float,
            upper_wick_ratio: float,
            lower_wick_ratio: float
    ) -> None:
        """
        Process a single bar for structure break detection.

        Note: This function follows the original pattern where momentum
        continuation is handled separately in the main loop.

        :param bar_index: Current bar index
        :param close: Close price
        :param high: High price
        :param low: Low price
        :param open_price: Open price (named to avoid shadowing 'open')
        :param atr_val: Current ATR value
        :param body_ratio: Body ratio of current candle
        :param is_bullish_body: True if candle has bullish body
        :param is_bearish_body: True if candle has bearish body
        :param trend: Current trend ('uptrend' or 'downtrend')
        :param prev_high: Previous bar's high price
        :param prev_low: Previous bar's low price
        :param swings: Dictionary of swing points
        :param active_levels: Active break levels dictionary
        :param builder: Result builder for recording signals
        :param config: Structure break configuration
        :param close_location: Close location within candle (0-1)
        :param upper_wick_ratio: Upper wick ratio
        :param lower_wick_ratio: Lower wick ratio
        """
        # Calculate minimum move threshold
        min_move = atr_val * config.min_break_atr_mult

        # Get break targets
        targets = BarProcessor._get_break_targets(swings, trend)

        # Check for gap breaks
        gap_breaks = BarProcessor._detect_gap_breaks(
            targets, open_price, min_move, bar_index
        )

        # Process potential new breaks
        BarProcessor._process_potential_breaks(
            targets=targets,
            gap_breaks=gap_breaks,
            bar_index=bar_index,
            close=close,
            high=high,
            low=low,
            body_ratio=body_ratio,
            is_bullish_body=is_bullish_body,
            is_bearish_body=is_bearish_body,
            close_location=close_location,
            upper_wick_ratio=upper_wick_ratio,
            lower_wick_ratio=lower_wick_ratio,
            atr_val=atr_val,
            min_move=min_move,
            config=config,
            active_levels=active_levels,
            builder=builder
        )

        # Update existing active levels
        keys_to_remove: Set[Tuple[str, int]] = set()

        for key, level in active_levels.items():
            should_remove, failure_signal = BarProcessor._update_active_level(
                level=level,
                bar_index=bar_index,
                close=close,
                high=high,
                low=low,
                prev_high=prev_high,
                prev_low=prev_low,
                config=config,
                builder=builder
            )

            if should_remove:
                keys_to_remove.add(key)
                if failure_signal:
                    builder.set_signal(failure_signal, bar_index)

        # Remove failed/completed levels
        for key in keys_to_remove:
            active_levels.pop(key, None)

        # Cleanup to prevent memory bloat
        BarProcessor._cleanup_active_levels(active_levels, config.max_active_levels)
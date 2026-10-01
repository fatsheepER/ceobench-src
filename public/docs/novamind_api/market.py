"""Market discovery and research tools."""

from typing import Dict, Optional
from . import _client


def research_market() -> Dict:
    """Attempt to discover a new customer segment.

    Costs $25K per attempt, 30% chance of discovering a random hidden group.

    Returns:
        Dict with status ('discovered', 'not_found', 'exhausted'), cost and
        remaining_undiscovered. discovered_group_id is present only on discovery.
    """
    return _client.call('research_market')


def research_group(group_id: str, target_level: Optional[int] = None) -> Dict:
    """Upgrade information level for a discovered group.

    Higher info levels give more accurate data about the group.

    Args:
        group_id: The group identifier (e.g., 'D_S01').
        target_level: Target info level (default: current + 1).

    Returns:
        Dict with research result.
    """
    args = {'group_id': group_id}
    if target_level is not None:
        args['target_level'] = target_level
    return _client.call('research_group', args)


def get_market_overview() -> Dict:
    """Get overview of all known customer segments.

    Returns:
        Dict with known_groups, undiscovered_count and macroeconomic (or None).
        Macroeconomic data includes measurement_day and publication_delay_days;
        measurement_day is the date of the reading, not its publication date.
    """
    return _client.call('get_market_overview')


def get_group_insights(group_id: str) -> Dict:
    """Get detailed insights for a specific customer group.

    Includes referral rates, reputation influence, segment characteristics.

    Args:
        group_id: The group identifier.

    Returns:
        Dict with group insight data and snapshot_day (survey date; 0 is valid).
        Network self_referral, outgoing and incoming rates are all in leads per
        1000 subscribers per day. A new read retrieves the same survey;
        research_group must complete to update it.
    """
    return _client.call('get_group_insights', {'group_id': group_id})

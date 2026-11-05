"""Analytics, monitoring, and operations tools."""

from typing import Dict, Optional
from . import _client


def get_social_posts(days: int = 7, limit: int = 50) -> Dict:
    """Get recent social media posts from customers.

    Args:
        days: Number of days to look back (default 7).
        limit: Maximum number of posts to return (default 50).

    Returns:
        Dict with social media post data.
    """
    return _client.call('get_social_posts', {'days': days, 'limit': limit})


def set_targeted_ops_spend(
    targeted_spend: Optional[Dict[str, float]] = None,
    *,
    by_group: Optional[Dict[str, float]] = None,
    by_plan: Optional[Dict[str, float]] = None,
    by_group_plan: Optional[Dict[str, Dict[str, float]]] = None,
    by_customer: Optional[Dict[str, float]] = None,
) -> Dict:
    """Set additional operations spend for four independent scopes.

    Each supplied mapping replaces its whole scope. Omitted or None arguments
    leave that scope unchanged. Passing {} clears only that scope. With no
    arguments, all scopes remain unchanged.

    Args:
        targeted_spend: Legacy positional alias for by_group. Supply either
            targeted_spend or by_group, not both.
        by_group: {group_id: daily_amount} dict.
        by_plan: {plan: daily_amount} dict. Plans are A, B, and C.
        by_group_plan: {group_id: {plan: daily_amount}} dict.
        by_customer: {customer_id_as_string: daily_amount} dict.

    Returns:
        Current settings and changes as {field: {before, after}}.
        Empty changes means unchanged; None marks an added or removed entry.
    """
    args = {'targeted_spend': targeted_spend}
    for name, value in (('by_group', by_group), ('by_plan', by_plan),
                        ('by_group_plan', by_group_plan), ('by_customer', by_customer)):
        if value is not None:
            args[name] = value
    return _client.call('set_targeted_ops_spend', args)


def set_targeted_dev_spend(targeted_spend: Dict[str, float]) -> Dict:
    """Set per-group targeted development spend.

    Per-group quality grows by 0.0225 × ln(1 + spend/5000) daily, where spend
    is that group's targeted development spend in USD per day. This adds to
    global development gains and persists after spending stops.

    Args:
        targeted_spend: {group_id: daily_amount} dict. Replaces the whole group
            mapping; omitted groups stop receiving targeted spend. {} clears it.

    Returns:
        Current settings and changes as {field: {before, after}}.
        Empty changes means unchanged; None marks an added or removed entry.
    """
    return _client.call('set_targeted_dev_spend', {'targeted_spend': targeted_spend})

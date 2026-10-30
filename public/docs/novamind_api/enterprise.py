"""Enterprise customer management tools."""

from typing import Dict, List
from . import _client


def send_enterprise_deal(deals: List) -> Dict:
    """Send enterprise deal offerings to one or more customers.

    Compact tuple format: deals=[[customer_id, [["plan", price_per_seat], ...]]]

    All contracts are month-to-month (1 month). Identify customers by customer_id,
    not message_id. This replies to an open negotiation. With no open thread,
    an active enterprise subscription is required to start renegotiation.
    Reply within 7 days of a customer message. At 7 days without a reply, a new
    lead is lost and an existing subscriber cancels.

    Args:
        deals: List of [customer_id, offerings] pairs.

    Returns:
        Dict with per-customer results when at least one deal succeeds. Failed
        entries contain error and customer_id when available, without success.

    Raises:
        NovaMindAPIError: If all deals fail.
    """
    return _client.call('send_enterprise_deal', {'deals': deals})


def reject_enterprise_deal(deals: List) -> Dict:
    """Reject enterprise negotiation threads.

    Identify customers by customer_id, not message_id. Rejection closes the open
    negotiation. A pending new lead is marked lost. An existing subscription
    remains unchanged.

    Args:
        deals: List of {'customer_id': int} dicts.

    Returns:
        Dict with per-customer results when at least one deal succeeds. Failed
        entries contain error and customer_id when available, without success.

    Raises:
        NovaMindAPIError: If all deals fail.
    """
    return _client.call('reject_enterprise_deal', {'deals': deals})

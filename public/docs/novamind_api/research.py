"""R&D research project tools."""

from typing import Dict
from . import _client


def start_research_project(tier: int) -> Dict:
    """Start an R&D research project.

    20 independent tiers, no dependencies. Use list_research_projects for each
    tier's cost, duration and quality distributions.
    Duration and quality boost are randomly sampled on start.

    Args:
        tier: Research tier (1-20).

    Returns:
        Dict with project start confirmation.
    """
    return _client.call('start_research_project', {'tier': tier})


def list_research_projects() -> Dict:
    """List all R&D research tiers and their status.

    Returns:
        Dict with tiers: costs, mean_days/std_days, mean_quality_boost/
        std_quality_boost, in_progress/completed counts, total_quality_boost,
        and projects (project_id, status, started_day, expected_completion_day,
        expected_quality_boost, remaining_days). Completed remaining_days is None.
    """
    return _client.call('list_research_projects')

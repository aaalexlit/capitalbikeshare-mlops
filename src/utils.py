from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_year_months(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> ([int], [[int]]):
    """Get list of months for each year."""
    assert (
        start_year <= end_year
    ), f"start_year must be less than or equal to end_year, {start_year} > {end_year}"
    assert (
        start_month >= 1 and start_month <= 12
    ), f"start_month must be between 1 and 12, not {start_month}"
    assert (
        end_month >= 1 and end_month <= 12
    ), f"end_month must be between 1 and 12, not {end_month}"
    assert (
        start_year >= 2018
    ), "monthly information is only available from 2018 onwards"
    if start_year == end_year:
        assert (
            start_month <= end_month
        ), f"start_month of the same year {start_year} must be less than or equal to end_month"
        return [start_year], [list(range(start_month, end_month + 1))]
    return list(range(start_year, end_year + 1)), [
        list(range(start_month, 13)),
        *[list(range(1, 13)) for _ in range(start_year + 1, end_year)],
        *[list(range(1, end_month + 1))],
    ]

from .settings import ClusterExpansionSettings

__all__ = ("settings_from_json",)


def settings_from_json(fname) -> ClusterExpansionSettings:
    """Initialize settings from JSON.

    Exists due to compatibility. You should instead use
    `ClusterExpansionSettings.load(fname)`

    Parameters:

    fname: str
        JSON file where settings are stored
    """
    return ClusterExpansionSettings.load(fname)  # pylint: disable=no-member

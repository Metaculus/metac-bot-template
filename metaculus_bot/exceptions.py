class UnitMismatchError(ValueError):
    """Raised when numeric percentiles appear mis-scaled relative to bounds.

    Used to bail on posting a prediction for a question while allowing the
    rest of the batch to continue.
    """

    pass

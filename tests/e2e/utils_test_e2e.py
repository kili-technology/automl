def debug_subprocess_pytest(result):
    import traceback

    if result.exception is not None:
        traceback.print_tb(result.exception.__traceback__)
    assert result.exception is None

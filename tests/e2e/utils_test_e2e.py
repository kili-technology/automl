def debug_subprocess_pytest(result):
    import traceback

    print(result.output)
    if result.exception is not None:
        traceback.print_tb(result.exception.__traceback__)
        print(result.exception)
    assert result.exit_code == 0

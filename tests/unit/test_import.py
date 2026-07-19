import tokenlens


def test_package_importable() -> None:
    assert isinstance(tokenlens.__version__, str)
    assert tokenlens.__version__ != ""

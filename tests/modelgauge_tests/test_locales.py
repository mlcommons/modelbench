import pytest

from modelgauge import locales


def test_is_valid():
    assert locales.is_valid("en_us")
    assert locales.is_valid("fr_fr")
    assert locales.is_valid("zh_cn")
    # this will fail and tell you if you forgot to update the list of supported locales
    assert not locales.is_valid("hi_in")
    assert not locales.is_valid("fake")


def test_display_for():
    assert locales.display_for(locales.EN_US) == "en_US"
    assert locales.display_for(locales.FR_FR) == "fr_FR"
    assert locales.display_for(locales.ZH_CN) == "zh_CN"
    assert locales.display_for(locales.HI_IN) == "hi_IN"
    assert locales.display_for("whatever") == "whatever"


def test_bad_locale():
    assert (
        locales.bad_locale("chocolate")
        == 'You requested "chocolate." Only en_us, fr_fr, zh_cn (in lowercase) are supported.'
    )


def test_validate_locale():
    with pytest.raises(AssertionError):
        locales.validate_locale("bad locale")
    assert locales.validate_locale("en_us")
    assert locales.validate_locale("fr_fr")
    assert locales.validate_locale("zh_cn")

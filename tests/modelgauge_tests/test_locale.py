import pytest

import modelgauge.locale as LOCALE

@pytest.mark.parametrize("loc,expected,strict,cased", [
        ("en_us", "en_US", True, True),
        ("en_US", "en_US", True, True),
        ("EN_US", "en_US", True, True),
        (" EN_US ", "en_US", True, True),
        ("fr_fr", "fr_FR", True, True),
        ("fr_FR", "fr_FR", True, True),
        ("FR_FR", "fr_FR", True, True),
        (" FR_FR ", "fr_FR", True, True),
        ("", None, True, True),
        (None, None, True, True),

        ("en_us", "en_us", True, False),
        ("en_US", "en_us", True, False),
        ("EN_US", "en_us", True, False),
        (" EN_US ", "en_us", True, False),
        ("fr_fr", "fr_fr", True, False),
        ("fr_FR", "fr_fr", True, False),
        ("FR_FR", "fr_fr", True, False),
        (" FR_FR ", "fr_fr", True, False),
        ("", None, True, False),
        (None, None, True, False),
    ])
def test_make(loc, expected, strict, cased):
    assert LOCALE.make(loc, strict, cased) == expected


@pytest.mark.parametrize("loc,expected,should_raise", [
        ("en_us", "en_us", False),
        ("en_US", "en_us", False),
        ("EN_US", "en_us", False),
        (" EN_US ", "en_us", False),
        ("fr_fr", "fr_fr", False),
        ("fr_FR", "fr_fr", False),
        ("FR_FR", "fr_fr", False),
        (" FR_FR ", "fr_fr", False),
        ("bad", "", True),
    ])
def test_make_defaults(loc, expected, should_raise):
    if should_raise:
        with pytest.raises(ValueError):
            LOCALE.make(loc)
    else:
        assert LOCALE.make(loc) == expected




@pytest.mark.parametrize("loc", ["bad", "en us"])
def test_make_strict(loc):
    with pytest.raises(ValueError):
        LOCALE.make(loc, strict=True)

@pytest.mark.parametrize("loc", ["bad", "en us"])
def test_make_lenient(loc):
    assert LOCALE.make(loc, strict=False) is None

def test_eq():
    assert LOCALE.eq(None, None)
    assert LOCALE.eq(None, "")
    assert LOCALE.eq("", "")
    assert LOCALE.eq("", None)
    assert LOCALE.eq("en_us", "en_US")
    assert LOCALE.eq("bad", "terrible")
    assert LOCALE.eq("bad", None)
    assert LOCALE.eq("bad", "")
    assert not LOCALE.eq("en_us", "fr_fr")
    assert not LOCALE.eq("fr_fr", "en_us")


@pytest.mark.parametrize("loc,expected", [
        ("en_us", True),
        ("en_US", True),
        ("EN_US", True),
        ("fr_fr", True),
        ("fr_FR", True),
        ("FR_FR", True),
        ("", False),
        (None, False),
        # these will fail and alert you if you forget to enable them when they're supported
        ("zh_CN", False),
        ("zh_cn", False),
        ("ZH_CN", False),
        ("hi_in", False),
        ("hi_IN", False),
        ("HI_IN", False),
    ])
def test_valid(loc:str, expected:bool):
    assert LOCALE.valid(loc) is expected

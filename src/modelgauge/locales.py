# Keep these in all lowercase
# Always and only use these named constants in function calls.
# They are meant to simplify the Locale(enum) and prevent case errors.
EN_US = "en_us"
FR_FR = "fr_fr"
ZH_CN = "zh_cn"
HI_IN = "hi_in"
DEFAULT_LOCALE = "en_us"

# add the other languages after we have official and practice prompt sets
LOCALES = (EN_US, FR_FR, ZH_CN)
# all the languages we have official and practice prompt sets for
PUBLISHED_LOCALES = (EN_US, FR_FR)


def is_valid(locale: str) -> bool:
    return locale in LOCALES


def display_for(locale: str) -> str:
    chunks = locale.split("_")
    try:
        assert len(chunks) == 2
        display = f"{chunks[0].lower()}_{chunks[1].upper()}"
    except:
        display = locale
    return display


def bad_locale(locale: str) -> str:
    return f"You requested \"{locale}.\" Only {', '.join(LOCALES)} (in lowercase) are supported."


def validate_locale(locale) -> bool:
    assert is_valid(locale), bad_locale(locale)
    return True

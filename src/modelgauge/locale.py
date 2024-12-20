# This replaces the Locale enum with a dumb string. The Locale enum was causing more issues than it was solving.

# Usage
# import modelgauge.locale as loc
# locale = loc.make(<your locale string>, cased=True|False)
# if cased is True, returns the standard case (en_US), else lowercase

LOCALES = {
    "en_us": "en_US",  # English, United States
    "fr_fr": "fr_FR",  # French, France
    # TODO: uncomment when we have prompt support for these locales
    # "zh_cn": "zh_CN",  # Simplified Chinese, China
    # "hi_in": "hi_IN" , # Hindi, India
}


def valid(loc: str) -> bool:
    if not loc:
        return False
    return LOCALES.get(loc.lower(), None) is not None


def eq(loc1: str, loc2: str) -> bool:
    return make(loc1, strict=False, cased=False) == make(loc2, strict=False, cased=False)


def make(loc: str | None, strict: bool = True, cased: bool = False) -> str | None:
    if not loc:
        return None

    loc_s = loc.strip().lower()
    loc_code = LOCALES.get(loc_s, None)

    if not loc_code:
        if strict:
            raise ValueError(f"Locale string {loc} is invalid.")
        else:
            return None

    return loc_code if cased else loc_code.lower()

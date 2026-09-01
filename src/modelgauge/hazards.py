class Hazards(list):
    """The hazard keys are as follows:

    - vcr: Violent Crimes
    - ncr: Non-Violent Crimes
    - src: Sex-Related Crimes
    - cse: Child Sexual Exploitation
    - dfm: Defamation
    - spc: Specialized Advice
    - prv: Privacy Violations
    - ipv: Intellection Property Violations
    - iwp: Indiscriminate Weapons (CBRNE)
    - hte: Hate
    - ssh: Suicide & Self-Harm
    - sxc: Sexual Content"""

    SUB_HAZARDS = {
        "spc": {"spc_ele", "spc_fin", "spc_hlt", "spc_lgl"},
        "sxc": {"sxc_prn"},
    }

    def __init__(self):
        super().__init__(["vcr", "ncr", "src", "cse", "dfm", "spc", "prv", "ipv", "iwp", "hte", "ssh", "sxc"])

    @property
    def _known_hazards(self):
        return set(self).union(*self.SUB_HAZARDS.values())

    def get_hazard_family_from_row(self, row: dict[str, str]) -> str:
        """Returns the first half of a two-part hazard string (e.g. "spc_ele" -> "spc")
        or the whole hazard string as-is if there's no _ (e.g. "vcr" -> "vcr")"""
        return row["hazard"].split("_")[0]

    def get_hazard_from_row(self, row: dict[str, str]) -> str:
        """Returns a hazard string as-is if it's valid (e.g. "vcr", "spc_ele", etc)"""
        hazard = row["hazard"]
        if hazard not in self._known_hazards:
            raise ValueError(
                f"Unknown hazard code {hazard!r} for prompt {row.get('release_prompt_id', '<unknown>')!r}."
            )
        return hazard

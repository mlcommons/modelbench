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

    def __init__(self):
        super().__init__(["vcr", "ncr", "src", "cse", "dfm", "spc", "prv", "ipv", "iwp", "hte", "ssh", "sxc"])

    @staticmethod
    def get_hazard_from_row(row):
        """Subhazards should all be grouped together"""
        return row["hazard"].split("_")[0]

from modelbench.hazards import HazardScore
from modelbench.scoring import LetterGradeMixin, NumericGradeMixin

fake_reference_standard = {
    "safe_cae_hazard-0.5": 0.99,
    "safe_cbr_hazard-0.5": 0.90,
    "safe_hat_hazard-0.5": 0.80,
    "safe_nvc_hazard-0.5": 0.70,
    "safe_scr_hazard-0.5": 0.60,
    "safe_ssh_hazard-0.5": 0.50,
    "safe_vcr_hazard-0.5": 0.40,
}


class FakeHazardScore:
    def __init__(self, key):
        self.key = key
        self.hazard_definition = FakeHazardDefinition(key)


class FakeHazardDefinition:
    def __init__(self, key):
        self.key = key

    def reference_standard(self):
        try:
            return fake_reference_standard[f"safe_{self.key}_hazard-0.5"]
        except:
            return 0.0


class FakeScoringClass(LetterGradeMixin, NumericGradeMixin):
    def __init__(self, hazard_score, value):
        self.hazard_score = hazard_score
        self.value = value

    def numeric_grade(self):
        return self._numeric_grade(self.hazard_score, self.value)


def test_numeric_grade():
    hazard_score = FakeHazardScore("cae")

    # is this correct or backwards?
    mixin_test = FakeScoringClass(hazard_score, 0.000001)
    assert mixin_test.numeric_grade() == 1
    assert mixin_test.text_grade() == "P"
    assert mixin_test.grade_label() == "Poor"

    mixin_test = FakeScoringClass(hazard_score, 0.999999)
    assert mixin_test.numeric_grade() == 5
    assert mixin_test.text_grade() == "E"
    assert mixin_test.grade_label() == "Excellent"

    mixin_test = FakeScoringClass(hazard_score, hazard_score.hazard_definition.reference_standard())
    assert mixin_test.numeric_grade() == 3
    assert mixin_test.text_grade() == "G"
    assert mixin_test.grade_label() == "Good"

    hazard_score = FakeHazardScore("ssh")
    mixin_test = FakeScoringClass(hazard_score, 1.51 * hazard_score.hazard_definition.reference_standard())
    assert mixin_test.numeric_grade() == 4
    assert mixin_test.text_grade() == "VG"
    assert mixin_test.grade_label() == "Very Good"

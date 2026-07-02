"""Guards for the calibration-point canonicalization notes (review 1.2-5).

Three (sigma, v0) points coexist in the repo -- (1.2005, 1.6219) pooled single
fit / (1.156, 1.681) radius-0.35 LOCO mean / (1.168, 1.712) radius-0.30 LOCO
mean = canonical. The DUT CLI default and the cruise CSV baseline are the
NON-canonical points, which is only safe as long as the cross-reference notes
naming all three survive edits. These tests pin the notes (docstrings + the
generated note file), so a refactor cannot silently drop the disambiguation.
"""
import examples.run_rq2_cruise_sensitivity as cruise
import examples.run_rq2_dut_validation as dut

CANONICAL = ("1.168", "1.712")
LOCO_R035 = ("1.156", "1.681")


def test_dut_docstring_names_all_three_points():
    doc = dut.__doc__
    for tok in CANONICAL + LOCO_R035 + ("1.20", "1.62"):
        assert tok in doc, f"DUT docstring lost calibration-point token {tok}"
    assert "CANONICAL" in doc


def test_cruise_docstring_names_all_three_points():
    doc = cruise.__doc__
    for tok in CANONICAL + LOCO_R035 + ("1.2005", "1.6219"):
        assert tok in doc, f"cruise docstring lost calibration-point token {tok}"
    assert "CANONICAL" in doc


def test_points_note_file_is_generated_with_all_three_points(tmp_path):
    path = cruise._write_points_note(tmp_path)
    assert path.name == "calibration_points_note.md"
    text = path.read_text(encoding="utf-8")
    for tok in CANONICAL + LOCO_R035 + ("1.2005", "1.6219"):
        assert tok in text
    assert "正準" in text

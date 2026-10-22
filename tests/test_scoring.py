from persona_holonomy.scoring import score_response


def test_genre_intrusion():
    theatrical = score_response("ACT IV: THE CATHEDRAL! Final boss. Quest unlocked!")
    ordinary = score_response("A parser reads the input structure, while a validator checks constraints.")
    assert theatrical["genre_intrusion"] == 1
    assert theatrical["persona_index"] > ordinary["persona_index"]
    assert theatrical["theatrical_headings"] == 1
    assert ordinary["persona_index"] == 0


def test_index_bounded():
    score = score_response("ACT IV: THE CATHEDRAL! Quest unlocked!!! 🐉")
    assert 0 <= score["persona_index"] <= 1

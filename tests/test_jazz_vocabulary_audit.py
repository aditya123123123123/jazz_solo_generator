from scripts.audit_jazz_vocabulary import audit_phrase, audit_phrases


def test_audit_phrase_counts_weak_beat_chromatic_approaches_to_chord_tones():
    phrase = {
        "solo_id": 1,
        "performer": "Test Player",
        "title": "Synthetic",
        "phrase_number": 1,
        "notes": [
            {"pitch": 60, "onset": 0.0, "duration": 0.25, "bar": 0, "beat": 1, "chord": "Cj7"},
            {"pitch": 63, "onset": 0.5, "duration": 0.25, "bar": 0, "beat": 1.5, "chord": ""},
            {"pitch": 64, "onset": 1.0, "duration": 0.25, "bar": 0, "beat": 2, "chord": ""},
        ],
    }

    row = audit_phrase(phrase)

    assert row["notes"] == 3
    assert row["analyzable_notes"] == 3
    assert row["weak_to_strong_pairs"] == 1
    assert row["chromatic_approaches"] == 1
    assert row["chromatic_approach_rate"] == 1.0


def test_audit_phrase_counts_two_note_enclosure_into_guide_tone():
    phrase = {
        "solo_id": 2,
        "performer": "Test Player",
        "title": "Synthetic",
        "phrase_number": 1,
        "notes": [
            {"pitch": 66, "onset": 0.0, "duration": 0.25, "bar": 0, "beat": 1, "chord": "C7"},
            {"pitch": 62, "onset": 0.5, "duration": 0.25, "bar": 0, "beat": 1.5, "chord": ""},
            {"pitch": 64, "onset": 1.0, "duration": 0.25, "bar": 0, "beat": 2, "chord": ""},
        ],
    }

    row = audit_phrase(phrase)

    assert row["enclosure_targets"] == 1
    assert row["enclosures"] == 1
    assert row["guide_tone_landings"] == 1
    assert row["guide_tone_landing_rate"] == 1 / 3


def test_audit_phrase_counts_beat_boundary_pickups_as_approaches():
    phrase = {
        "solo_id": 3,
        "performer": "Test Player",
        "title": "Synthetic",
        "phrase_number": 1,
        "notes": [
            {"pitch": 61, "onset": 0.8, "duration": 0.15, "bar": 0, "beat": 1, "chord": "Cj7"},
            {"pitch": 60, "onset": 1.0, "duration": 0.25, "bar": 0, "beat": 2, "chord": ""},
        ],
    }

    row = audit_phrase(phrase)

    assert row["weak_to_strong_pairs"] == 1
    assert row["chromatic_approaches"] == 1


def test_audit_phrases_aggregates_by_performer_and_overall():
    phrases = [
        {
            "solo_id": 1,
            "performer": "A",
            "title": "One",
            "phrase_number": 1,
            "notes": [
                {"pitch": 63, "onset": 0.5, "duration": 0.25, "bar": 0, "beat": 1.5, "chord": "Cj7"},
                {"pitch": 64, "onset": 1.0, "duration": 0.25, "bar": 0, "beat": 2, "chord": ""},
            ],
        },
        {
            "solo_id": 2,
            "performer": "B",
            "title": "Two",
            "phrase_number": 1,
            "notes": [
                {"pitch": 60, "onset": 0.0, "duration": 0.25, "bar": 0, "beat": 1, "chord": "F7"},
            ],
        },
    ]

    report = audit_phrases(phrases)

    assert report["overall"]["phrases"] == 2
    assert report["overall"]["chromatic_approaches"] == 1
    assert report["overall"]["chromatic_approach_rate"] == 1.0
    assert {row["performer"] for row in report["by_performer"]} == {"A", "B"}

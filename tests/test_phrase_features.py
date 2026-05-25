from pathlib import Path

from src.generation.phrase_features import PhraseFeature, load_phrase_features


def test_load_phrase_features_aggregates_by_phrase_token(tmp_path: Path):
    csv_path = tmp_path / "phrase_clusters_all.csv"
    csv_path.write_text(
        "solo_id,performer,phrase_number,length_beats,num_notes,note_density,pitch_mean,pitch_range,pitch_std,rest_ratio,contour,starts_on_downbeat,phrase_duration,phrase_token\n"
        "1,A,1,4,8,2.0,60,10,1,0.10,ascending,True,2,PHRASE_00\n"
        "2,B,1,8,12,1.5,64,14,1,0.30,descending,False,4,PHRASE_00\n"
        "3,C,1,6,10,1.7,62,12,1,0.20,descending,False,3,PHRASE_00\n"
        "4,D,1,2,4,2.0,55,6,1,0.05,flat,True,1,PHRASE_01\n"
    )

    features = load_phrase_features(csv_path)

    assert set(features) == {"PHRASE_00", "PHRASE_01"}
    phrase_00 = features["PHRASE_00"]
    assert isinstance(phrase_00, PhraseFeature)
    assert phrase_00.median_length_beats == 6.0
    assert phrase_00.median_num_notes == 10
    assert phrase_00.median_density == 1.7
    assert phrase_00.median_pitch_mean == 62.0
    assert phrase_00.median_pitch_range == 12.0
    assert phrase_00.median_rest_ratio == 0.2
    assert phrase_00.contour == "descending"
    assert phrase_00.starts_on_downbeat is False


def test_default_phrase_features_loads_real_cluster_tokens():
    features = load_phrase_features()

    assert "PHRASE_00" in features
    assert "PHRASE_63" in features
    assert len(features) == 64

"""
Module for analyzing speaking time and participation
"""
def analyze_participation(annotated_data):
    """
    Returns analytics on speaking time and participation by gender and speaker.
    annotated_data: list of dicts [{"speaker", "start_time", "end_time", "annotation", "gender"}]
    Returns: dict with per-speaker and per-gender stats
    """
    import pandas as pd
    df = pd.DataFrame(annotated_data)
    # Calculate speaking time per utterance
    df["speaking_time"] = df.apply(lambda row: float(row["end_time"]) - float(row["start_time"]), axis=1)
    # Aggregate by speaker
    speaker_stats = df.groupby("speaker")["speaking_time"].sum().to_dict()
    # Aggregate by gender
    gender_stats = df.groupby("gender")["speaking_time"].sum().to_dict()
    # Participation ratio
    total_time = df["speaking_time"].sum()
    participation = {s: t/total_time if total_time else 0 for s, t in speaker_stats.items()}
    return {
        "speaker_stats": speaker_stats,
        "gender_stats": gender_stats,
        "participation_ratio": participation,
        "total_time": total_time
    }
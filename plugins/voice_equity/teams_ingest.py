"""
Module for ingesting Microsoft Teams meeting data
"""
def ingest_teams_data(source):
    """
    Ingests meeting data from Microsoft Teams.
    Supports transcript file (.json, .csv) or API (Graph API).
    Returns: list of dicts [{"speaker": str, "start_time": str, "end_time": str, "text": str, "demographics": dict}]
    """
    import os, json, csv
    data = []
    if isinstance(source, str) and os.path.isfile(source):
        if source.endswith('.json'):
            with open(source, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # Example Teams transcript format
            for entry in raw.get('transcript', []):
                data.append({
                    "speaker": entry.get("speaker"),
                    "start_time": entry.get("start_time"),
                    "end_time": entry.get("end_time"),
                    "text": entry.get("text"),
                    "demographics": entry.get("demographics", {})
                })
        elif source.endswith('.csv'):
            with open(source, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append({
                        "speaker": row.get("speaker"),
                        "start_time": row.get("start_time"),
                        "end_time": row.get("end_time"),
                        "text": row.get("text"),
                        "demographics": json.loads(row.get("demographics", "{}")) if row.get("demographics") else {}
                    })
        else:
            raise ValueError("Unsupported file format. Use .json or .csv.")
    elif isinstance(source, dict):
        # Assume API response format
        for entry in source.get('transcript', []):
            data.append({
                "speaker": entry.get("speaker"),
                "start_time": entry.get("start_time"),
                "end_time": entry.get("end_time"),
                "text": entry.get("text"),
                "demographics": entry.get("demographics", {})
            })
    else:
        raise ValueError("Source must be a file path or API response dict.")
    return data

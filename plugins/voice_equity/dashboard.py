"""
Module for generating voice equity dashboards
"""
def generate_dashboard(analytics):
    """
    Generates visual dashboard from analytics data.
    Shows bar chart for speaker participation and pie chart for gender split.
    """
    import matplotlib.pyplot as plt
    # Speaker participation bar chart
    speakers = list(analytics["speaker_stats"].keys())
    times = list(analytics["speaker_stats"].values())
    plt.figure(figsize=(10,5))
    plt.bar(speakers, times, color='skyblue')
    plt.title('Speaking Time by Participant')
    plt.xlabel('Speaker')
    plt.ylabel('Total Speaking Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Gender split pie chart
    if analytics.get("gender_stats"):
        labels = list(analytics["gender_stats"].keys())
        values = list(analytics["gender_stats"].values())
        plt.figure(figsize=(7,7))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Speaking Time by Gender')
        plt.tight_layout()
        plt.show()
    """
    Generates visual dashboard from analytics data.
    Shows bar chart for speaker participation and pie chart for demographic split.
    """
    import matplotlib.pyplot as plt
    # Speaker participation bar chart
    speakers = list(analytics["speaker_stats"].keys())
    times = list(analytics["speaker_stats"].values())
    plt.figure(figsize=(10,5))
    plt.bar(speakers, times, color='skyblue')
    plt.title('Speaking Time by Participant')
    plt.xlabel('Speaker')
    plt.ylabel('Total Speaking Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Demographic split pie chart
    if analytics["demographic_stats"]:
        labels = [f"{k[0]}: {k[1]}" for k in analytics["demographic_stats"].keys()]
        values = list(analytics["demographic_stats"].values())
        plt.figure(figsize=(7,7))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Speaking Time by Demographic')
        plt.tight_layout()
        plt.show()

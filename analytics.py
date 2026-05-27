import matplotlib.pyplot as plt


def plot_study_vs_score(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["studytime"], df["G3"], alpha=0.7)
    ax.set_title("Study Time vs Final Score")
    ax.set_xlabel("Study Time")
    ax.set_ylabel("Final Score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_failures_impact(df):
    failures_impact = df.groupby("failures")["G3"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    failures_impact.plot(kind="bar", ax=ax)
    ax.set_title("Failures Impact on Final Score")
    ax.set_xlabel("Past Failures")
    ax.set_ylabel("Average Final Score")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_absences_impact(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["absences"], df["G3"], alpha=0.7)
    ax.set_title("Absences vs Final Score")
    ax.set_xlabel("Absences")
    ax.set_ylabel("Final Score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

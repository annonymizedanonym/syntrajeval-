import pandas as pd


def trr_score(df: pd.DataFrame):
    """
    Time reverse ratio: make sure that the time is moving forward
    """
    try:
        TRR_score = []
        for tid in df["tid"].unique():
            trr = 0
            traj = df[df["tid"] == tid]  
            for i in range(len(traj) - 1):
                if traj["day"].iloc[i + 1] == traj["day"].iloc[i]:
                    if traj["hour"].iloc[i + 1] < traj["hour"].iloc[i]:
                        trr += 1
                        print(i, traj)

                    else:
                        continue

                if traj["day"].iloc[i + 1] > traj["day"].iloc[i]:
                    continue
            TRR_score.append(trr / (len(traj) - 1))
        return 1 - sum(TRR_score)  # make best score 1

    except KeyError:
        return None

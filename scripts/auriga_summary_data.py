import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("data/iza_et_al_2022/table_1.csv")
    df = df.drop(columns=["Galaxy", "Group"])
    text = "AVERAGES"
    text += "\n" + str("--------")
    text += "\n" + str(df.mean())
    text += "\n\n" + str("STANDARD DEVIATIONS")
    text += "\n" + str("-------------------")
    text += "\n" + str(df.std())

    with open("data/iza_et_al_2022/table_1_summary.txt", "w") as f:
        f.write(text)

import pandas as pd

# Load dataset
play_tennis = pd.read_csv("play_tennis.csv")

def naive_bayes(outlook='Sunny', temp='Mild', humidity='High', wind='Weak'):
    cases = ["Yes", "No"]
    results = []

    for play in cases:
        # Prior
        y = (play_tennis['play'].value_counts()[play]) / (play_tennis['play'].value_counts().sum())

        # Conditional probabilities with Laplace smoothing
        c1 = ((pd.crosstab(play_tennis['outlook'], play_tennis['play'])[play][outlook] + 1) /
              (pd.crosstab(play_tennis['outlook'], play_tennis['play'])[play].sum() + len(play_tennis['outlook'].unique())))
        
        c2 = ((pd.crosstab(play_tennis['temp'], play_tennis['play'])[play][temp] + 1) /
              (pd.crosstab(play_tennis['temp'], play_tennis['play'])[play].sum() + len(play_tennis['temp'].unique())))
        
        c3 = ((pd.crosstab(play_tennis['humidity'], play_tennis['play'])[play][humidity] + 1) /
              (pd.crosstab(play_tennis['humidity'], play_tennis['play'])[play].sum() + len(play_tennis['humidity'].unique())))
        
        c4 = ((pd.crosstab(play_tennis['wind'], play_tennis['play'])[play][wind] + 1) /
              (pd.crosstab(play_tennis['wind'], play_tennis['play'])[play].sum() + len(play_tennis['wind'].unique())))
        
        arg = y * c1 * c2 * c3 * c4
        results.append(arg)

    if results[0] >= results[1]:
        return f'Yes, You Can Play Tennis. P(Yes)={results[0]:.5f}, P(No)={results[1]:.5f}'
    else:
        return f'No, You Cannot Play Tennis. P(Yes)={results[0]:.5f}, P(No)={results[1]:.5f}'

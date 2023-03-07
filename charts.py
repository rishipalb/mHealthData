import matplotlib.pyplot as plt
import seaborn as sns

def barchart(x,y,title):
    """
    draws bar plot,
    x: feature importance or ranking
    y: labels
    """
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=x,y=y, orient='h').set(title=title)
    return fig
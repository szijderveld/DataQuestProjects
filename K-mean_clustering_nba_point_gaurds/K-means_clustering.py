import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


nba = pd.read_csv("nba_2013.csv")
point_guards = nba[nba['pos'] == 'PG']


#point per game
point_guards['ppg'] = point_guards['pts'] / point_guards['g']

# Sanity check, make sure ppg = pts/g
point_guards[['pts', 'g', 'ppg']].head(5)


point_guards = point_guards[point_guards['tov'] != 0]
#assit turn over ratio
point_guards['atr'] = point_guards['ast'] / point_guards['tov']

# plt.scatter(point_guards['ppg'], point_guards['atr'], c='y')
# plt.title("Point Guards")
# plt.xlabel('Points Per Game', fontsize=13)
# plt.ylabel('Assist Turnover Ratio', fontsize=13)
# plt.show()


#identified 5 general regions in the data that point gaurds fell into
num_clusters = 5
#  random function to generate a list, length: num_clusters, of indices
random_initial_points = np.random.choice(point_guards.index, size=num_clusters)
#  random indices to create the centroids
centroids = point_guards.loc[random_initial_points]


plt.scatter(point_guards['ppg'], point_guards['atr'], c='yellow')
plt.scatter(centroids['ppg'], centroids['atr'], c='red')
plt.title("Centroids")
plt.xlabel('Points Per Game', fontsize=13)
plt.ylabel('Assist Turnover Ratio', fontsize=13)
plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(point_guards[['ppg', 'atr']])
point_guards['cluster'] = kmeans.labels_



def visualize_clusters(df, num_clusters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for n in range(num_clusters):
        clustered_df = df[df['cluster'] == n]
        plt.scatter(clustered_df['ppg'], clustered_df['atr'], c=colors[n-1])
        plt.xlabel('Points Per Game', fontsize=13)
        plt.ylabel('Assist Turnover Ratio', fontsize=13)
    plt.show()
visualize_clusters(point_guards, num_clusters)
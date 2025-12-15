from __init__ import *
from params import *

xg_arr_10 = pd.read_csv("xg_stats_rs10.csv").to_numpy()
xg_arr_2 = pd.read_csv("xg_stats_rs2.csv").to_numpy()
df22 = pd.read_csv("xg_stats_rs22.csv")
columns = df22.columns.tolist()
xg_arr_22 = df22.to_numpy()

xg_arr = np.vstack([xg_arr_10,
                    xg_arr_2,
                    xg_arr_22])


goals = np.zeros((Length, Width), dtype=float)
attempts = np.zeros_like(goals)
xg_plot = np.zeros_like(goals)
for i in range(xg_arr.shape[0]):
    # go throught the grid an calculate an xg associated w/ each grid point
    I = min(int(xg_arr[i,0]),Length-1) # x-index
    J = min(int(xg_arr[i,1]),Width-1) # x-index

    # always incriment attempts array, only incriment goals if scored
    attempts[I,J] += 1
    scored = xg_arr[i,3]
    if scored: goals[I,J] += 1

# plot
mask = attempts != 0
xg_plot[mask] = goals[mask] / attempts[mask]

plt.figure(figsize=(8,4))
plt.imshow(
    xg_plot.T,          
    origin='lower',     
    aspect='auto',
    extent=[0, Length, 0, Width],
    cmap='plasma',
)
plt.colorbar()
plt.xlabel('Length')
plt.ylabel('Width')
plt.title('XG MAP')
plt.savefig("expected_goals_heatmap.png")

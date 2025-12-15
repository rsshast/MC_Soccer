from __init__ import *
from params import *

xg_arr_2 = pd.read_csv("xg_stats_rs2.csv").to_numpy()
xg_arr_5 = pd.read_csv("xg_stats_rs5.csv")
xg_arr_6 = pd.read_csv("xg_stats_rs6.csv")
xg_arr_7 = pd.read_csv("xg_stats_rs7.csv")
xg_arr_10 = pd.read_csv("xg_stats_rs10.csv").to_numpy()
xg_arr_14 = pd.read_csv("xg_stats_rs14.csv").to_numpy()
xg_arr_16 = pd.read_csv("xg_stats_rs16.csv").to_numpy()
xg_arr_19 = pd.read_csv("xg_stats_rs19.csv").to_numpy()
xg_arr_28 = pd.read_csv("xg_stats_rs28.csv").to_numpy()
xg_arr_48 = pd.read_csv("xg_stats_rs48.csv").to_numpy()
xg_arr_50 = pd.read_csv("xg_stats_rs50.csv").to_numpy()
xg_arr_56 = pd.read_csv("xg_stats_rs56.csv").to_numpy()
xg_arr_66 = pd.read_csv("xg_stats_rs66.csv").to_numpy()
xg_arr_99 = pd.read_csv("xg_stats_rs99.csv").to_numpy()
df22 = pd.read_csv("xg_stats_rs22.csv")
columns = df22.columns.tolist()
xg_arr_22 = df22.to_numpy()

xg_arr = np.vstack([xg_arr_10,
                    xg_arr_2,
                    xg_arr_6,
                    xg_arr_7,
                    xg_arr_14,
                    xg_arr_16,
                    xg_arr_19,
                    xg_arr_22,
                    xg_arr_28,
                    xg_arr_5,
                    xg_arr_50,
                    xg_arr_56,
                    xg_arr_48,
                    xg_arr_99,
                    xg_arr_66])
mask = np.all(xg_arr[:,:2] >= 0, axis=1)
xg_arr = xg_arr[mask]
print(xg_arr.shape[0])

goals = np.zeros((Length, Width), dtype=float)
attempts = np.zeros_like(goals)
xg_plot = np.zeros_like(goals)
for i in range(xg_arr.shape[0]):
    # go throught the grid an calculate an xg associated w/ each grid point
    I = min(int(xg_arr[i,0]),Length-1) # x-index
    J = min(int(xg_arr[i,1]),Width-1) # x-index
    if I < Length / 2 + 1: 
        ifoo = 99 - I
        I = ifoo

    # always incriment attempts array, only incriment goals if scored
    attempts[I,J] += 1
    scored = xg_arr[i,3]
    if scored: goals[I,J] += 1

# plot
mask = attempts != 0
xg_plot[mask] = goals[mask] / attempts[mask]

subset = xg_plot[70:, 5:45].T  
plt.figure()
plt.imshow(subset, origin='lower', aspect='auto', cmap='plasma')
plt.colorbar()
plt.xlabel('Length')
plt.ylabel('Width')
plt.title('XG MAP')
plt.savefig("expected_goals_heatmap.png")

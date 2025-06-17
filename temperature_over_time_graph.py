from ase.io import iread
import matplotlib.pyplot as plt
import numpy as np

temps = []
timesteps = 1000
timestep = 0.5
i = 0

for atoms in iread("output/CH4_md.xyz", index=":"):
    if i > timesteps:
        break

    temps.append(atoms.get_temperature())
    i += timestep

t = np.arange(0, timesteps + 0.5, timestep)

fig, ax = plt.subplots()
ax.plot(t, temps)

ax.set(xlabel="Time (fs)", ylabel="Temperature (K)",
       title="Burning of H2 gas")
ax.grid()

fig.savefig("output/CH4_md.png")

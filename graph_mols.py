from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import RDLogger
import argparse
import matplotlib.pyplot as plt
import numpy as np

RDLogger.DisableLog('rdApp.*')  # supress hydrogen warnings

parser = argparse.ArgumentParser(description="Graph molecule quantity over time")
parser.add_argument("-i", "--input_reaction", type=str, default="CH4", help="Input reaction name")
parser.add_argument("-p", action="store_true", help="Flag for graphing percentages of molecules")
args = parser.parse_args()

input_file = f"output/{args.input_reaction}_md.xyz.species"
input_log = f"output/{args.input_reaction}_md.log"
log_interval = 1    # assume log_interval of 1

timestep = 0
mol_count = {}
with open(input_file, 'r') as f:
    for line in f:
        line = line.split()
        timestep = int(line[1].strip(':'))
        info = line[2:]
        for mol in mol_count:
            mol_count[mol].append(0)    # assume 0 count of all molecules
        for i in range(0, len(info), 2):
            smiles = info[i]
            num = int(info[i+1])
            mol = CalcMolFormula(Chem.MolFromSmiles(smiles))
            if mol in mol_count:
                mol_count[mol][-1] = num    # update last value if molecule in dict
            else:
                mol_count[mol] = [0]*timestep + [num]   # create list for molecule if not in dict

log_size = 1
with open(input_log, 'r') as f:
    f.readline()
    f.readline()
    ts_size = float(f.readline().split()[0])
    log_size += len(f.readlines())
log_size *= log_interval
traj_interval = log_size / timestep
time = np.arange(0, (timestep+1) * ts_size * traj_interval, ts_size * traj_interval)

if args.p:
    max_num_mols = 0
    for mol in mol_count:
        max_num_mols = max(max_num_mols, max(mol_count[mol]))
    for mol in mol_count:
        mol_count[mol] = [i / max_num_mols * 100 for i in mol_count[mol]]

for mol in mol_count:
    plt.plot(time, mol_count[mol], label=mol)

print(f"Graphing based on traj_interval: {traj_interval}, log_interval: {log_interval}, and {timestep*traj_interval} timesteps of {ts_size} ps each.")
plt.title(f"{args.input_reaction}K Reaction Over {max(time)} Picoseconds")
plt.xlabel("Time (ps)")
plt.ylabel("Number of Molecules (%)") if args.p else plt.ylabel("Number of Molecules")
plt.legend()
plt.show()

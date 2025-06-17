from utils import generate_configuration
from mace.calculators import mace_mp

mp = mace_mp(model='large', default_dtype='float64', device='cuda')

smiles_strings = ["C", "O=O"]
counts = [50, 100]
dimensions = [15, 30, 30]
conformers = 25
temperature = 298
reaction_name = "H2O"
outfile = f"{reaction_name}_{temperature}.xyz"

# generate_configuration(smiles_strings, counts, mp, dimensions, conformers, temperature, outfile)
left = generate_configuration([smiles_strings[0]], [counts[0]], mp, dimensions, conformers, temperature)
right = generate_configuration([smiles_strings[1]], [counts[1]], mp, dimensions, conformers, temperature)
right.translate([15, 0, 0])
left.extend(right)

left.write(outfile)
print(f"{reaction_name}_{temperature}.xyz written")

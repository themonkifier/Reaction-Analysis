from ase.io import read
import os

atoms = read("output/CH4_md.xyz", index=":")

for conf in atoms:
    conf.wrap()
    conf.write("output/CH4_md_wrapped.xyz", append=True)

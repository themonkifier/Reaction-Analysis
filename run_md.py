# from orb_models.forcefield import pretrained
# from orb_models.forcefield.calculator import ORBCalculator
from utils import run_md_simulation
import torch
from mace.calculators import mace_mp
import argparse
import ase
from ase import units


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MD simulation")
    parser.add_argument("-i", "--input_file", type=str, help="Input configuration file")
    parser.add_argument("-d", "--output_dir", type=str, help="Output directory", default="output")
    parser.add_argument("-o", "--output_file", type=str, help="Output trajectory file", default=None)
    parser.add_argument("-c", "--cell_size", type=float, help="Cell box side length")
    parser.add_argument("-t", "--temperature", type=float, help="Desired system temperature (Kelvin)")
    parser.add_argument("-s", "--timestep", type=float, help="Timestep (fs)", default=0.5)
    parser.add_argument("-f", "--friction", type=float, help="Friction (1/fs)", default=0.01)
    parser.add_argument("--total_steps", type=int, help="Total number of steps taken", default=2000)
    parser.add_argument("--traj_interval", type=int, help="Interval (# timesteps) to write atomic positions to output trajectory", default=20)
    parser.add_argument("--log_interval", type=int, help="Interval (# timesteps) to log system characteristics", default=1)
    args = parser.parse_args()

    device = get_device()
    mp = mace_mp(model="large", default_dtype="float64", device=device)

    run_md_simulation(mp, args.input_file, args.output_dir, args.output_file, args.cell_size,
    args.temperature, args.timestep * ase.units.fs, args.friction / ase.units.fs, args.total_steps, args.traj_interval, args.log_interval)

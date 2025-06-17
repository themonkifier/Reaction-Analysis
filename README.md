Run the md simulation: `python3 run_md.py -i CH4.xyz -c 30 -t 850 -s 0.1 -f 0.01 --total_steps 25000 --traj_interval 100 --log_interval 1`
Wrap according to PBC (for visualization): `python3 wrap_xyz.py`
Generate temperature over time graph: `python3 temperature_over_time_graph.py`
Run reaction analysis (this outputs a .html file): `reacnetgenerator -t xyz -i output/CH4_md.xyz -a C H O -c 30 0 0 0 30 0 0 0 30`

import numpy as np
import csv
from ase.io import read
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.calculators.dftd3 import DFTD3
from ase.parallel import parprint
from gpaw import GPAW, PW
from functions.geometry import strain, interlayer_distance, get_shifts
from functions.structure import create_bilayer
from functions.bandstructure import calc_gap
from functions.util import repeate_cells
from scipy.interpolate import LinearNDInterpolator

# Get structure
path = '../../structures/MoS2-WSe2-MatterSim/1.05_3027/structure_ml.json'
atoms = read(path)

# Read strain data
data = np.genfromtxt('band_edges_medium.csv', skip_header=1,
                     dtype=float, delimiter=',')
strain_arr = (data[:, 0] - 1)*100
MoS2_homo = data[:, 1]
MoS2_lumo = data[:, 2]
WSe2_homo = data[:, 3]
WSe2_lumo = data[:, 4]

# Get geometric parameters
x_MoS2, y_MoS2, MoS2_strain = strain(atoms, 'Mo')  # len = 525
x_WSe2, y_WSe2, WSe2_strain = strain(atoms, 'W')  # the rest are len = 484
x_dist, y_dist, interlayer_dist = interlayer_distance(atoms)

shift_dict = get_shifts(atoms)
x_shifts = shift_dict['shifts'][:, 0]
y_shifts = shift_dict['shifts'][:, 1]


parameters = {'x': x_WSe2,
              'y': y_WSe2,
              'WSe2_strain': WSe2_strain,
              'x_shift': x_shifts,
              'y_shift': y_shifts,
              'interlayer_distance': interlayer_dist}

# Calculate the band gap from distance and shifts
raw_gap = []
lumo_levels = []
homo_levels = []


for i in range(len(x_WSe2)):
    bilayer = create_bilayer(parameters['interlayer_distance'][i],
                             lattice_length=3.2515,
                             a_shift=parameters['x_shift'][i],
                             b_shift=parameters['y_shift'][i])

    calc = GPAW(mode=PW(500),
                xc='PBE',
                kpts={'size': (8, 8, 1)},
                txt=None)
    d3_calc = DFTD3(dft=calc)
    bilayer.calc = d3_calc

    # Constrain the metal 
    c = FixAtoms(indices=[atom.index for atom in bilayer
                          if (atom.symbol == 'Mo' or atom.symbol == 'W')])
    bilayer.set_constraint(c)

    opt = BFGS(bilayer,
               trajectory=f'traj_files/opt_{x_WSe2[i]:.2f}_{y_WSe2[i]:.2f}')
    # run the optimization until forces are smaller than fmax
    opt.run(fmax=0.01)

    gap, lumo, homo = calc_gap(bilayer, kpts=30)
    raw_gap.append(gap)
    lumo_levels.append(lumo)
    homo_levels.append(homo)

    parprint(f'Cell {i} out of {len(x_WSe2)} relaxed')


# Linear interpolation for the Mo strain
x_MoS2_large, y_MoS2_large, MoS2_strain_large = repeate_cells(x_MoS2, y_MoS2,
                                                              MoS2_strain,
                                                              range(-1, 2),
                                                              atoms.cell[0],
                                                              atoms.cell[1])
MoS2_strain_intp = LinearNDInterpolator(list(zip(x_MoS2_large, y_MoS2_large)),
                                        MoS2_strain_large)


# Linear interpolation for the strain corrections
data = np.genfromtxt("band_edges_medium.csv", dtype=float,
                     skip_header=1, delimiter=',')

strain_arr = data[:, 0]
MoS2_homo = data[:, 1]
MoS2_lumo = data[:, 2]
WSe2_homo = data[:, 3]
WSe2_lumo = data[:, 4]

strain_mesh_x, strain_mesh_y = np.meshgrid(strain_arr, strain_arr,
                                           indexing="ij")
MoS2_LUMO_grid, WSe2_HOMO_grid = np.meshgrid(MoS2_lumo, WSe2_homo,
                                             indexing="ij")
gap_grid = MoS2_LUMO_grid - WSe2_HOMO_grid

strain_x_arr = strain_mesh_x.ravel()
strain_y_arr = strain_mesh_y.ravel()
gap_arr = gap_grid.ravel()

strain_correction_interp = LinearNDInterpolator(list(zip(strain_x_arr,
                                                         strain_y_arr)),
                                                gap_arr)

# Reference value for gap correction
ref_gap = MoS2_lumo[-1] - WSe2_homo[0]
parprint(f"Reference gap is {ref_gap:.2f}eV")


corrected_gap = []
gap_correction_arr = []
# Correct the gap values with the strain correction
for i in range(len(x_WSe2)):
    # Get the MoS2 strain
    WSe2_strain_val = parameters['WSe2_strain'][i]

    x_val = parameters['x'][i]
    y_val = parameters['x'][i]

    MoS2_strain_val = MoS2_strain_intp([x_val], [y_val])[0]

    new_gap = strain_correction_interp([MoS2_strain_val], [WSe2_strain_val])[0]
    gap_correction = new_gap - ref_gap

    corrected_gap.append(raw_gap[i] + gap_correction)
    gap_correction_arr.append(gap_correction)


x = parameters["x"]
y = parameters["y"]
inter_dist = parameters["interlayer_distance"]


with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    # write header
    writer.writerow(["x", "y", "interlayer_distance",
                     "corrected_gap", "correction"])
    # write each row
    for i in range(len(x)):
        writer.writerow([x[i], y[i], inter_dist[i],
                         corrected_gap[i], gap_correction_arr[i]])

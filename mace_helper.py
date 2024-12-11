import numpy as np
import pandas as pd


def get_mace_xyz_str(
    atoms,
    forces,
    energy,
):
    cell = atoms.cell.array
    res = f"{len(atoms)}\n"
    res += f'Energy={energy} Lattice="{cell[0,0]} {cell[0,1]} {cell[0,2]} {cell[1,0]} {cell[1,1]} {cell[1,2]} {cell[2,0]} {cell[2,1]} {cell[2,2]}" Properties=species:S:1:pos:R:3:forces:R:3\n'
    symbols = np.array(atoms.symbols)
    pos = atoms.positions
    for i in range(len(forces)):
        res += f"{symbols[i]} {pos[i,0]} {pos[i,1]} {pos[i,2]} {forces[i,0]} {forces[i,1]} {forces[i,2]}\n"

    return res


def MACE_DF(
    path,
    pot_file_name="MACE_model.model_float_cuda-lammps.pt",
    elements=["Al", "Cu", "Zr"],
    no_domain_decomposition=True,
):
    elements_str = ""
    for e in elements:
        elements_str += f" {e}"

    if pot_file_name is None:
        fname = path
        pot_file_name = path.split("/")[-1]
    else:
        fname = f"{path}/{pot_file_name}"

    if no_domain_decomposition:
        ps_str = "pair_style mace no_domain_decomposition\n"
    else:
        ps_str = "pair_style mace\n"

    config = [
        f"{ps_str}",
        f"pair_coeff * * {pot_file_name} {elements_str}\n",
    ]
    files = [fname]

    pot = pot = pd.DataFrame(
        {
            "Name": "MACE",
            "Filename": [files],
            "Model": ["Custom"],
            "Species": [elements],
            "Config": [config],
        }
    )

    return pot

import shutil

import ase
import ase.db
import numpy as np
import pandas as pd
from ase.units import Bohr, Ha

eVperA_in_HaperBohr = Bohr / Ha
AinBohr = 1 / Bohr
eVinHa = 1 / Ha


def ase_atoms_to_runner_str(
    atoms: ase.Atoms,
    energy: float,
    forces: np.ndarray,
    atomic_charge: np.ndarray = None,
    total_charge: float = 0.0,
    comment: str = "None",
) -> str:
    cell = atoms.cell.copy()
    cell *= AinBohr
    positions = atoms.positions.copy()
    positions *= AinBohr
    forces = forces.copy()
    forces *= eVperA_in_HaperBohr
    energy *= eVinHa
    symbols = np.array(atoms.symbols)
    if atomic_charge is None:
        atomic_charge = np.full(len(symbols), 0.0, dtype=float)
    s = "begin\n"
    s += f"comment {comment}\n"
    for v in cell:
        s += f"lattice    {v[0]}  {v[1]}  {v[2]}\n"

    if len(positions) != len(forces):
        raise ValueError("Forces and positions do not have the same length")
    for i in range(len(positions)):
        s += f"atom {positions[i, 0]} {positions[i, 1]} {positions[i, 2]} {symbols[i]} {atomic_charge[i]} 0.0 {forces[i, 0]} {forces[i, 1]} {forces[i, 2]}\n"
    s += f"energy {energy}\n"
    s += f"charge {total_charge}\n"
    s += "end\n"
    return s


def ase_atoms_to_n2p2_str(
    atoms: ase.Atoms,
    energy: float,
    forces: np.ndarray,
    atomic_charge: np.ndarray = None,
    total_charge: float = 0.0,
    comment: str = "None",
) -> str:
    cell = atoms.cell.copy()
    positions = atoms.positions.copy()
    forces = forces.copy()
    symbols = np.array(atoms.symbols)
    if atomic_charge is None:
        atomic_charge = np.full(len(symbols), 0.0, dtype=float)
    s = "begin\n"
    s += f"comment {comment}\n"
    for v in cell:
        s += f"lattice    {v[0]}  {v[1]}  {v[2]}\n"

    if len(positions) != len(forces):
        raise ValueError("Forces and positions do not have the same length")
    for i in range(len(positions)):
        s += f"atom {positions[i, 0]} {positions[i, 1]} {positions[i, 2]} {symbols[i]} {atomic_charge[i]} 0.0 {forces[i, 0]} {forces[i, 1]} {forces[i, 2]}\n"
    s += f"energy {energy}\n"
    s += f"charge {total_charge}\n"
    s += "end\n"
    return s


def radial_func(eta, Rij, Rs=0):
    return np.exp(-eta * (Rij - Rs) ** 2)


def dgdr(r, eta, cutoff):
    return np.exp(-eta * r**2.0) * (
        -2.0 * eta * r * (0.5 * (np.cos(np.pi * r / cutoff) + 1.0))
        - np.pi / (2.0 * cutoff) * np.sin(cutoff * r / cutoff)
    )


def create_radial_functions(elements, dmin, n_functions, cutoff):
    eta = np.zeros(n_functions)
    rmax = cutoff / 2.0
    rangeturn = rmax - dmin
    interval = rangeturn / (n_functions - 1)

    delta = 0.0001
    deltar = 0.0001

    rturn = [rmax]
    for i in range(1, n_functions):
        rturn.append(rmax - i * interval)
    rturn = np.array(rturn)
    f_str = ""
    for i in range(1, n_functions):
        etatrial = eta[i]
        rtrial = np.arange(0, rmax, 0.001)
        while np.all(rtrial > rturn[i]):
            dgdr_vals = dgdr(eta=etatrial, r=rtrial, cutoff=cutoff)

        eta[i] = etatrial
        f_str += "test"


def NNP_DF(
    path,
    elements,
    cutoff,
    mode="RuNNer",
    cflength=1.8897261328,
    cfenergy=0.0367493254,
    showew=False,
    showewsum=0,
    maxew=0,
    resetew=False,
    epoch=None,
    remote=False,
):
    for ele in elements:
        atomic_number = ase.data.atomic_numbers[ele]
        atomic_number = f"{atomic_number:03d}"
        # if os.path.exists(f"{path}/weights.{atomic_number}.data"):
        #    continue

        if mode.lower() == "runner":
            if epoch is None:
                shutil.copy(
                    f"{path}/optweights.{atomic_number:03d}.out",
                    f"{path}/weights.{atomic_number:03d}.data",
                )
            else:
                shutil.copy(
                    f"{path}/{epoch:06d}.short.{atomic_number:03d}.out",
                    f"{path}/weights.{atomic_number:03d}.data",
                )

        elif mode.lower() == "n2p2":
            if epoch.lower() == "pass":
                pass
            elif isinstance(epoch, int):
                shutil.copy(
                    f"{path}/weights.{atomic_number:03d}.{epoch:06d}.out",
                    f"{path}/weights.{atomic_number:03d}.data",
                )
            else:
                raise ValueError("Set epoch to 'pass' or int with epoch to use.")

    elements_str = ""
    for e in elements:
        elements_str += f" {e}"

    # cutoff not be smaller than cutoff in bohr, but can be slightly larger
    # so add a small number
    # cutoff = cutoff * cflength + 1e-8

    if showew:
        showew = "yes"
    else:
        showew = "no"

    if resetew:
        resetew = "yes"
    else:
        resetew = "no"

    if remote:
        files = ['input.nn', 'scaling.data']
        for ele in elements:
            atomic_number = ase.data.atomic_numbers[ele]
            files.append(f'weights.{atomic_number:03d}.data')
        fname = [[f'{path}/{f}' for f in files]]
        dirpath = './'
                
    else:
        fname = ['']
        dirpath = path

    pot = pd.DataFrame(
        {
            "Name": "NNP_{elements_str}",
            "Filename": fname,
            "Model": ["Custom"],
            "Species": [elements],
            "Config": [
                [
                    f"pair_style hdnnp {cutoff} dir {dirpath} showew {showew} showewsum {showewsum} cflength {cflength} cfenergy {cfenergy} maxew {maxew} resetew {resetew}\n",
                    f"pair_coeff * * {elements_str}\n",
                ]
            ],
        }
    )
    return pot

import ase.db
import numpy as np
import pandas as pd


def get_energy(row):
    return row.nist_normed_energy


def get_forces(row):
    return row.data["dft_forces"]


def get_stresses(row):
    return None


def atoms_to_cfg(atoms, symbol2index, forces, **kwargs):
    cfg_str = "BEGIN_CFG\n" " Size\n" f" {len(atoms)}\n" f" Supercell\n"
    for v in atoms.cell.array:
        cfg_str += f"   {v[0]}\t{v[1]}\t{v[2]}\n"

    cfg_str += " AtomData: id type cartes_x cartes_y cartes_z fx fy fz\n"
    symbols = np.array(atoms.symbols)
    positions = atoms.positions
    for i in range(len(atoms)):
        X = symbol2index[symbols[i]]
        p = positions[i]
        f = forces[i]
        cfg_str += f"\t\t {i+1}\t{X}\t{p[0]}\t{p[1]}\t{p[2]}\t{f[0]}\t{f[1]}\t{f[2]}\n"

    elem_map = ""
    for k, v in symbol2index.items():
        elem_map += f"{k}={v}, "
    cfg_str += f"Feature\t Element order: {elem_map}\n"
    comments = ""
    for k, v in kwargs.items():
        comments += f"Feature\t {k}: {v}\n"
    cfg_str += comments
    cfg_str += "END_CFG\n\n"
    return cfg_str


def row_to_cfg(
    row,
    symbol2index,
    energy_function=None,
    force_function=None,
    stress_function=None,
    **kwargs,
):
    cfg_str = "BEGIN_CFG\n" " Size\n" f" {row.natoms}\n" f" Supercell\n"
    for v in row.cell:
        cfg_str += f"   {v[0]}\t{v[1]}\t{v[2]}\n"

    cfg_str += " AtomData: id type cartes_x cartes_y cartes_z fx fy fz\n"

    if energy_function is None:
        energy = row.free_energy
    else:
        energy = energy_function(row)

    if force_function is None:
        forces = row.forces
    else:
        forces = force_function(row)

    if stress_function is None:
        stresses = row.stress
    else:
        stresses = stress_function(row)

    symbols = row.symbols
    positions = row.positions
    for i in range(row.natoms):
        X = symbol2index[symbols[i]]
        p = positions[i]
        f = forces[i]
        cfg_str += f"\t\t {i+1}\t{X}\t{p[0]}\t{p[1]}\t{p[2]}\t{f[0]}\t{f[1]}\t{f[2]}\n"

    cfg_str += " Energy\n"
    cfg_str += f"  {energy}\n"
    if stresses is not None:
        vir = -row.volume * stresses
        cfg_str += " PlusStress: xx yy zz yz xz xy\n"
        cfg_str += f"\t{vir[0]}\t{vir[1]}\t{vir[2]}\t{vir[3]}\t{vir[4]}\t{vir[5]}\n"

    elem_map = ""
    for k, v in symbol2index.items():
        elem_map += f"{k}={v}, "
    cfg_str += f"Feature\t Element order: {elem_map}\n"
    comments = ""
    for k, v in kwargs.items():
        comments += f"Feature\t {k}: {v}\n"
    cfg_str += comments
    cfg_str += "END_CFG\n\n"
    return cfg_str


def cfg_to_db(file, index2symbol, db):
    with open(file, "r") as fd:
        read_atoms = False
        read_energy = False
        read_size = False
        read_supercell = False
        read_stress = False
        for line in fd:
            if "BEGIN_CFG" in line:
                positions = []
                forces = []
                energy = 0
                vir = None
                volume = 0
                cell = []
                symbols = []
                features = {}
                size = 0
                continue

            elif "END_CFG" in line:
                atoms = ase.Atoms(
                    positions=positions, cell=cell, symbols=symbols, pbc=True
                )
                data = {
                    "forces": np.array(forces, dtype=float),
                }
                if vir is not None:
                    volume = np.abs(np.linalg.det(cell))
                    # stresses = vir / -volume
                    # data["stresses"] = stresses
                    data["vir"] = vir
                data.update(dct_features)
                db.write(atoms, data=data, cfg_energy=energy, **kw_features)
                continue

            elif "Size" in line:
                read_size = True
                continue

            elif "AtomData" in line:
                read_atoms = True
                n_atoms = 0
                continue

            elif "Supercell" in line:
                read_supercell = True
                cell_lines = 0
                continue

            elif "Energy" in line:
                read_energy = True
                continue

            elif "PlusStress:" in line:
                read_stress = True
                continue

            elif "Feature" in line:
                dct_features = {}
                kw_features = {}
                tmp = line.strip().split()
                val = tmp[2:]
                if len(val) == 1:
                    try:
                        val = float(val[0])
                    except:
                        val = val[0]
                    kw_features[tmp[1]] = val

                else:
                    dct_features[tmp[1]] = val
                continue

            elif read_size:
                size = int(line.strip())
                read_size = False
                continue

            elif read_supercell:
                cell_lines += 1
                vec = line.strip().split()
                vec = np.array(vec, dtype=float)
                cell.append(vec)
                if cell_lines == 3:
                    read_supercell = False
                    cell = np.array(cell)
                continue

            elif read_atoms:
                n_atoms += 1
                tmp = line.strip().split()
                n = tmp[0]
                symbols.append(index2symbol[tmp[1]])
                positions.append(np.array(tmp[2:5]))
                forces.append((np.array(tmp[5:8])))
                if n_atoms == size:
                    read_atoms = False
                continue

            elif read_energy:
                energy = float(line.strip())
                read_energy = False
                continue

            elif read_stress:
                vir = np.array(line.split(), dtype=float)
                read_stress = False
                continue


def MTP3_DF(
    path,
    pot_file="output.mtp",
    elements=["Si", "O", "C"],
    interactive=False,
    extrapolation_control=False,
    extrapolation_break=10,
    extrapolation_save=2,
    extrapolation_configs="out/preselected.cfg",
):
    elements_str = ""
    for e in elements:
        elements_str += f" {e}"

    if extrapolation_control:
        extrapolation_control = "true"
    else:
        extrapolation_control = "false"

    pot = pd.DataFrame(
        {
            "Name": "MTP",
            "Filename": [[f"{path}/{pot_file}"]],
            "Model": ["Custom"],
            "Species": [elements],
            "Config": [
                [
                    f"pair_style mlip load_from={pot_file} extrapolation_control={extrapolation_control} extrapolation_control:threshold_break={extrapolation_break} extrapolation_control:threshold_save={extrapolation_save} extrapolation_control:save_extrapolative_to={extrapolation_configs}\n",
                    "pair_coeff * *\n",
                ]
            ],
        }
    )
    return pot


def MTP_DF(
    path,
    pot_file="output.mtp",
    ini_file=None,
    elements=["Si", "O", "C"],
    interactive=False,
):
    if ini_file is None:
        if interactive:
            default_ini_str = f"mtp-filename {path}/{pot_file}\n" "select FALSE\n"
        else:
            default_ini_str = f"mtp-filename {pot_file}\n" "select FALSE\n"
        ini_file = "pyiron_default_mlip.ini"
        with open(f"{path}/{ini_file}", "w") as f:
            f.write(default_ini_str)

    elements_str = ""
    for e in elements:
        elements_str += f" {e}"

    pot = pd.DataFrame(
        {
            "Name": "MTP",
            "Filename": [
                [
                    f"{path}/{pot_file}",
                    f"{path}/{ini_file}",
                ]
            ],
            "Model": ["Custom"],
            "Species": [elements],
            "Config": [
                [
                    f"pair_style mlip {ini_file}\n",
                    "pair_coeff * *\n",
                ]
            ],
        }
    )
    return pot


def cfgs_with_min_dist(f1, f2, d):
    with open(f1, "r") as f1, open(f2, "w") as f2:
        s = False
        w = False
        lines = []
        n1 = 0
        n2 = 0
        for l in f1:
            if w:
                w = False
                if mindist >= d:
                    n2 += 1
                    f2.writelines(lines)
                lines.clear()

            if s:
                lines.append(l)
                if "Feature   mindist" in l:
                    mindist = float(l.split()[2])

                elif "END_CFG" in l:
                    s = False
                    w = True

            elif "BEGIN_CFG" in l:
                n1 += 1
                s = True
                lines.append(l)

        print(f"Wrote {n2} of {n1} structure to new file")


def batch_cfgs(f, batchsize=1000000):
    with open(f, "r") as f1:
        batch_n = 0
        s = False
        w = False
        lines = []
        nStructs = 0
        f2 = open(f"{f}_batch{batch_n}", "w")
        for l in f1:
            if nStructs >= batchsize:
                nStructs = 0
                batch_n += 1
                f2.close()
                f2 = open(f"{f}_batch{batch_n}", "w")

            if w:
                nStructs += 1
                w = False
                f2.writelines(lines)
                lines.clear()

            if s:
                lines.append(l)
                if "END_CFG" in l:
                    s = False
                    w = True

            elif "BEGIN_CFG" in l:
                s = True
                lines.append(l)

        print(f"Wrote {batch_n+1} structure files")
        f2.close()

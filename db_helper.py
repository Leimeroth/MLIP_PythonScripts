import itertools

import ase
import ase.data
import ase.db
import ase.units
import numpy as np
import pandas as pd
from pyiron_atomistics import ase_to_pyiron

import pace_helper as pah

AlCuZr_nist_dict = {
    "Cu": -0.2490765974999998,
    "Zr": -2.2370127159999997,
    "Al": -0.35200846750000014,
}

SiOC_single_atom_dict = {
    "C": -0.42483456,  # free energy
    "O": -0.74034035,
    "Si": -4.14567113,
}

CuZr_lat_name_dict = {
    "mp_30603": "Cu5Zr",
    "mp_1216441": "Cu51Zr14",  # stable
    "mp_1188040": "Cu3Zr",
    "mp_1195821": "Cu8Zr3",  # stable
    "mp_1072655": "Cu2Zr",
    "mp_1188077": "Cu10Zr7",  # stable
    "mp_2210": "CuZr-B2",
    "mp_1080022": "CuZr-B33",
    "mp_1067210": "CuZr-B19'",
    "mp_193": "CuZr2-t",  # stable
    "mp_583800": "CuZr2-c",
    "mp_1077372": "CuZr2-m",
    "mp_580287": "CuZr3",
}

CuZr_nice_lat_name_dict = {
    "mp_30603": "Cu$_{5}$Zr",
    "mp_1216441": "Cu$_{51}$Zr$_{14}$",
    "mp_1188040": "Cu$_{3}$Zr",
    "mp_1195821": "Cu$_{8}$Zr$_{3}$",
    "mp_1072655": "Cu$_{2}$Zr",
    "mp_1188077": "Cu$_{10}$Zr$_{7}$",
    "mp_2210": "CuZr-B2",
    "mp_1080022": "CuZr-B33",
    "mp_1067210": "CuZr-B19'",
    "mp_193": "CuZr$_{2}$-t",
    "mp_583800": "CuZr$_{2}$-c",
    "mp_1077372": "CuZr$_{2}$-m",
    "mp_580287": "CuZr$_{3}$",
}

CuZr_PD_phases = {
    "mp_30603": "Cu5Zr",
    "mp_1216441": "Cu51Zr14",
    "mp_1195821": "Cu8Zr3",
    "mp_1188077": "Cu10Zr7",
    "mp_2210": "CuZr-B2",
    "mp_193": "CuZr2-t",
}

MIXING_ENERGY_DICT = {
    6: -9.9360649028125,  # Diamond is more stable, but use graphite here
    8: -5.428510635,  # DFT energy from linus
    14: -10.00968854125,  # Si diamond DFT energy
    13: -3.39244442,
    29: -3.4851639225000004,
    40: -6.240364759000001,
}


AMU_IN_GRAMM = 1.6605390666e-24
ANGSTROM_IN_CM = 1e-8
DENSITY_FACTOR = AMU_IN_GRAMM / ANGSTROM_IN_CM**3

ACE514_MIXING_ENERGY_DICT = {
    29: -3.48915999365505,
    40: -6.23959670426115,
}

MENDELEV2019_MIXING_ENERGY_DICT = {
    29: -3.283116209166775,
    40: -6.4692658936244,
}


def get_composition(atoms):
    numbers = atoms.numbers
    nTot = len(numbers)
    composition = {}
    for n in np.unique(numbers):
        sym = ase.data.chemical_symbols[n]
        nEle = len(numbers[numbers == n])
        composition[sym] = nEle / nTot
    return composition


def weight_to_atomic(weight_composition):
    molTot = 0
    mols = {}
    for ele, w in weight_composition.items():
        nEle = ase.data.atomic_numbers[ele]
        mass = ase.data.atomic_masses[nEle]
        mol = w / mass
        mols[ele] = mol
        molTot += mol

    atomic_composition = {}
    for ele, mol in mols.items():
        atomic_composition[ele] = mol / molTot

    return atomic_composition


def atomic_to_weight(atomic_composition):
    wTot = 0
    weights = {}
    for ele, f in atomic_composition.items():
        nEle = ase.data.atomic_numbers[ele]
        mass = ase.data.atomic_masses[nEle]
        wEle = f * mass
        weights[ele] = wEle
        wTot += wEle

    weight_composition = {}
    for ele, wEle in weights.items():
        weight_composition[ele] = wEle / wTot

    return weight_composition


def tag_elements(db):
    with db:
        for row in db.select():
            if row.get("Elements") is None:
                numbers = np.unique(row.numbers)
                ele_str = ""
                for n in numbers:
                    sym = ase.data.chemical_symbols[n]
                    ele_str += sym
                db.update(row.id, Elements=ele_str)


def tag_lattice(db, lat_pos):
    with db:
        for row in db.select():
            if row.get("lattice") is None:
                path = row.path.split("/")
                lat = path[lat_pos]
                db.update(row.id, lattice=lat)


def sort_structures(db, force_key="dft_forces"):
    ids = []
    for row in db.select():
        if not np.all(np.array(row.symbols) == np.sort(row.symbols)):
            ids.append(row.id)

    with db:
        for row in db.select():
            if row.id in ids:
                atoms = row.toatoms()
                numbers = atoms.get_atomic_numbers()
                sort_array = np.argsort(numbers)
                atoms = atoms[sort_array]
                data = row.data
                data[force_key] = row.data[force_key][sort_array]
                db.update(id=row.id, atoms=atoms, data=data)
    return


def normalize_energy_struct(energy, symbols, subtract_energy_dict):
    for ele, val in subtract_energy_dict.items():
        n_ele = len(symbols[symbols == ele])
        energy -= n_ele * val
    return energy


def normalize_energy(
    db,
    subtract_energy_dict=AlCuZr_nist_dict,
    energy_key="dft_energy",
    normed_energy_key="nist_normed_energy",
):
    with db:
        for row in db.select():
            if row.get(normed_energy_key) is None:
                symbols = np.array(row.symbols)
                e = row.get(energy_key)
                if e is None:
                    raise ValueError(f"{energy_key} does not exist for row {row.id}")
                e = normalize_energy_struct(e, symbols, subtract_energy_dict)
                db.update(row.id, **{normed_energy_key: e})
    return


def convex_hull_dist(db, energy="nist_normed_energy", update=False):
    ase_atoms = []
    energy_corrected_per_atom = []
    NUMBER_OF_ATOMS = []
    db_ids = []
    for row in db.select():
        ase_atoms.append(row.toatoms())
        energy_corrected_per_atom.append(row[energy] / row.natoms)
        NUMBER_OF_ATOMS.append(row.natoms)
        db_ids.append(row.id)

    df = pd.DataFrame(
        {
            "ase_atoms": ase_atoms,
            "energy_corrected_per_atom": energy_corrected_per_atom,
            "NUMBER_OF_ATOMS": NUMBER_OF_ATOMS,
            "db_ids": db_ids,
        }
    )

    pah.compute_convexhull_dist(df)
    if update:
        with db:
            for row in db.select():
                dbid = row.id
                dfrow = df.loc[df.db_ids == dbid]
                e_chull = dfrow.e_chull_dist_per_atom.values[0]
                if e_chull < 0.0:
                    e_chull = 0.0
                db.update(dbid, dist_from_chull=e_chull)

    else:
        with db:
            for row in db.select():
                if row.get("dist_from_chull") is None:
                    dbid = row.id
                    dfrow = df.loc[df.db_ids == dbid]
                    e_chull = dfrow.e_chull_dist_per_atom.values[0]
                    if e_chull < 0.0:
                        e_chull = 0.0
                    db.update(dbid, dist_from_chull=e_chull)
    return


def min_distance(db):
    with db:
        for row in db.select():
            min_distance = row.get("min_distance")
            if min_distance is not None:
                continue
            s = ase_to_pyiron(row.toatoms())
            n = s.get_neighbors(1)
            d = np.min(n.distances)
            db.update(row.id, min_distance=d)
    return


def min_distance_per_Element(db):
    with db:
        for row in db.select():
            numbers = row.numbers
            s = ase_to_pyiron(row.toatoms())
            n = s.get_neighbors(1)
            # s_indices = np.arange(len(s))
            distances = n.distances
            min_ds = {}
            for elen in np.unique(numbers):
                sym = ase.data.chemical_symbols[elen]
                ele_filter = numbers == elen
                ele_distances = distances[ele_filter]
                d = np.min(ele_distances)
                min_ds[f"{sym}_min_distance"] = d
            db.update(row.id, **min_ds)
    return


def max_force(db, forces="forces"):
    with db:
        for row in db.select():
            if row.get("max_force") is None:
                f = row.data[forces]
                max_f = np.max(np.linalg.norm(f, axis=1))
                db.update(row.id, max_force=max_f)


def min_distance_per_atom(db):
    with db:
        for row in db.select():
            if "min_distances" in row.data.keys():
                continue
            s = ase_to_pyiron(row.toatoms())
            n = s.get_neighbors(1)
            min_distances = n.distances.flatten()
            db.update(row.id, data={"min_distances": min_distances})
    return


def max_min_distance(db):
    with db:
        for row in db.select():
            if row.get("max_min_distance") is not None:
                continue
            min_distances = row.data["min_distances"]
            db.update(row.id, max_min_distance=np.max(min_distances))


def set_densities(db):
    with db:
        for row in db.select():
            if row.get("density") is None:
                density = row.mass / row.volume * DENSITY_FACTOR
                db.update(row.id, density=density)


def get_formation_enthalpy(energy, numbers, energy_dict=MIXING_ENERGY_DICT):
    natoms = len(numbers)
    eq_energy = 0
    for n in np.unique(numbers):
        nele = len(numbers[numbers == n])
        eele = MIXING_ENERGY_DICT[n]
        eq_energy += nele * eele
    return (energy - eq_energy) / natoms


def set_formation_enthalpies(
    db, energy_key="nist_normed_energy", energy_dict=MIXING_ENERGY_DICT, overwrite=False
):
    if not overwrite:
        with db:
            for row in db.select():
                if row.get("formation_enthalpy") is None:
                    fe = get_formation_enthalpy(
                        energy=row[energy_key], numbers=row.numbers
                    )
                    db.update(row.id, formation_enthalpy=fe)
    else:
        with db:
            for row in db.select():
                fe = get_formation_enthalpy(energy=row[energy_key], numbers=row.numbers)
                db.update(row.id, formation_enthalpy=fe)


def get_double_atoms(db):
    dbl_lst = []
    ats_lst = [(row.id, row.toatoms()) for row in db.select()]
    for t1, t2 in itertools.combinations(ats_lst, 2):
        if t1[1] == t2[1]:
            dbl_lst.append((t1[0], t2[0]))
    return dbl_lst


def delete_double_atoms(db):
    dbl_lst = get_double_atoms(db)
    if len(dbl_lst) > 0:
        del_lst = np.unique(np.array(dbl_lst)[:, 1]).tolist()
        db.delete(del_lst)


def get_str_list_matrix(symbols):
    symbol_matrix = [[s1 + s2 for s1 in symbols] for s2 in symbols]
    symbol_matrix = np.array(symbol_matrix)
    return symbol_matrix


def min_distance_per_element_combination(db):
    with db:
        for row in db.select():
            a = row.toatoms()
            distance_matrix = a.get_all_distances(mic=True)
            distance_matrix[distance_matrix == 0] = 10000
            symbols = a.get_chemical_symbols()
            sym_combinations = [
                s1 + s2
                for s1, s2 in itertools.combinations_with_replacement(
                    np.unique(symbols), 2
                )
            ]
            symbol_matrix = get_str_list_matrix(symbols)
            min_d_dict = {
                sym: np.min(distance_matrix[symbol_matrix == sym])
                for sym in sym_combinations
            }
            db.update(row.id, **min_d_dict)


def name_phases(db, lat_name_dict, selection=None, filter_func=None, **select_kwargs):
    with db:
        for row in db.select(selection=selection, filter=filter_func, **select_kwargs):
            try:
                lat = row.get("lattice")
                cname = lat_name_dict[lat]
                db.update(row.id, common_name=cname)
            except KeyError:
                print(f"{lat} not found in lat_name_dict")


def get_concentration_sort_array(
    structures,
    ele_number=40,
):
    numbers = [s.numbers for s in structures]
    counts = np.array([len(n[n == ele_number]) for n in numbers])
    lens = np.array([(len(n) for n in numbers)])
    concentrations = counts / lens
    return np.argsort(concentrations)

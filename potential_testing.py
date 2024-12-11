import hashlib as hl
import os
import subprocess

import ase.calculators.lammpslib
import ase.db
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pyiron_atomistics import Project, ase_to_pyiron
    from pyiron_atomistics.lammps.structure import UnfoldingPrism

    import db_helper as dbh
    import pyiron_helper as pyih
except ImportError:
    pass



import mlip_helper as mlh

# from pyiron_base import Settings

ENERGY_PER_ATOM_LABEL = "Energy [eV/atom]"

EDFT_MINUS_ECHULL_LABEL = r"E$_{\mathrm{DFT}}$-E$_{\mathrm{chull}}$ [eV/atom]"
EIP_MINUS_EDFT_LABEL = r"E$_{\mathrm{IP}}$-E$_{\mathrm{DFT}}$  [eV/atom]"
EDFT_LABEL = r"E$_{\mathrm{DFT}}$ [eV/atom]"
EIP_LABEL = r"E$_{\mathrm{IP}}$ [eV/atom]"
FDFT_LABEL = r"F$_{\mathrm{DFT}}$ [eV/$\mathrm{\AA}$]"
FIP_LABEL = r"F$_{\mathrm{IP}}$ [eV/$\mathrm{\AA}$]"
FDIFF_LABEL = r"F$_{\mathrm{DFT}}$ - F$_{\mathrm{IP}}$ [eV/$\mathrm{\AA}$]"
FDIFF_ABS_LABEL = r"|F$_{\mathrm{DFT}}$ - F$_{\mathrm{IP}}$| [eV/$\mathrm{\AA}$]"

cubic_lattices = ["fcc", "bcc", "sc", "diamond"]
crystal_lattices = ["fcc", "bcc", "sc", "diamond", "hcp"]
hexagonal_lattices = ["hcp"]

diffCu_nist = -0.2490765974999998
diffZr_nist = -2.2370127159999997
diffAl_nist = -0.35200846750000014

CoreRadii = {
    "Al": 1.0054367731,
    "Cu": 1.2171076727,
    "Zr": 1.587531747,
}

elements_lattices = {
    "Al": "fcc",
    "Cu": "fcc",
    "Zr": "hcp",
}

ele_latticeTuple = {
    "Al": ("fcc", "hcp", "bcc"),
    "Cu": ("fcc", "hcp", "bcc"),
    "Zr": ("hcp", "fcc", "bcc"),
}


def _get_fig_ax(fig_ax=None, figsize=(7, 7)):
    if fig_ax is None:
        return plt.subplots(figsize=figsize)
    return fig_ax


class StandardPlot_DFs:
    def __init__(self, dft_DF, pot_DF, diff_DF=None):
        self.dft_DF = dft_DF
        self.pot_DF = pot_DF
        if diff_DF is None:
            self.diff_DF = get_diff_DF(dft_DF=dft_DF, pot_DF=pot_DF)
        else:
            self.diff_DF = diff_DF


class EF_StandardPlots:
    def __init__(
        self,
        dft_DF,
        pot_DF,
        testmarker='^',
        diff_DF=None,
        test_dft_DF=None,
        test_pot_DF=None,
        test_diff_DF=None,
        ids=None,
        test_ids=None,
    ):
        self.dft_DF = dft_DF
        self.pot_DF = pot_DF
        if diff_DF is None:
            self.diff_DF = get_diff_DF(self.dft_DF, self.pot_DF)
        else:
            self.diff_DF = diff_DF

        if test_dft_DF is not None:
            self.test_dft_DF = test_dft_DF
            self.test_pot_DF = test_pot_DF
            if test_diff_DF is None:
                self.test_diff_DF = get_diff_DF(test_dft_DF, test_pot_DF)

        if ids is not None:
            self.dft_DF = self.dft_DF[self.dft_DF.index.isin(ids)]
            self.pot_DF = self.pot_DF[self.pot_DF.index.isin(ids)]
            self.diff_DF = self.diff_DF[self.diff_DF.index.isin(ids)]

        if test_ids is not None:
            self.test_dft_DF = self.test_dft_DF[self.test_dft_DF.index.isin(test_ids)]
            self.test_pot_DF = self.test_pot_DF[self.test_pot_DF.index.isin(test_ids)]
            self.test_diff_DF = self.test_diff_DF[
                self.test_diff_DF.index.isin(test_ids)
            ]
        self.figsize = (7, 7)
        self.testmarker=testmarker

    def plot_E_over_convex_hull(self, test=False, fig_ax=None):
        fig, ax = _get_fig_ax(fig_ax, self.figsize)

        ax.scatter(
            self.dft_DF.chull_dist,
            self.diff_DF.energy_per_atom,
            edgecolor="black",
            facecolor="none",
            s=20,
            alpha=0.4,
        )
        if test:
            ax.scatter(
                self.test_dft_DF.chull_dist,
                self.test_diff_DF.energy_per_atom,
                edgecolor="red",
                facecolor="none",
                s=20,
                alpha=0.4,
                marker=self.testmarker,
            )
            ax.legend(["train", "test"])
        ax.set_xlabel(EDFT_MINUS_ECHULL_LABEL)
        ax.set_ylabel(EIP_MINUS_EDFT_LABEL)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(-0.1, 0.1)
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def get_cumulated_RMSEs(dft_DF, diff_DF, dists):
        RMSEs = []
        for d in dists:
            db_ids = dft_DF.index[dft_DF.chull_dist < d]
            e = diff_DF.loc[db_ids].energy_per_atom.values
            RMSEs.append(np.sqrt(np.mean(e**2)))
        return RMSEs

    def plot_cumulated_E_RMSE(self, dists=np.arange(0.2, 10.1, 0.2), test=False):
        fig, ax = plt.subplots(figsize=(7, 7))
        RMSEs = self.get_cumulated_RMSEs(
            dft_DF=self.dft_DF, diff_DF=self.diff_DF, dists=dists
        )
        ax.scatter(
            dists,
            RMSEs,
            edgecolor="black",
            facecolor="black",
            s=10,
        )
        if test:
            tRMSEs = self.get_cumulated_RMSEs(
                dft_DF=self.test_dft_DF, diff_DF=self.test_diff_DF, dists=dists
            )
            ax.scatter(
                dists,
                tRMSEs,
                edgecolor="red",
                facecolor="red",
                s=10,
                marker=self.testmarker,
            )
            ax.legend(["train", "test"])
        ax.set_xlabel(EDFT_MINUS_ECHULL_LABEL)
        ax.set_ylabel("Cumulated RMSE [eV/atom]")
        # ax.set_xlim(0, 1)
        # ax.set_ylim(-0.1, 0.1)
        fig.tight_layout()
        return fig, ax

    def plot_energy_scatter(self, test=False, fig_ax=None):
        fig, ax = _get_fig_ax(fig_ax, self.figsize)
        ax.scatter(
            self.dft_DF.energy_per_atom,
            self.pot_DF.energy_per_atom,
            edgecolor="black",
            facecolor="none",
            s=20,
            alpha=0.4,
        )
        if test:
            ax.scatter(
                self.test_dft_DF.energy_per_atom,
                self.test_pot_DF.energy_per_atom,
                edgecolor="red",
                facecolor="none",
                s=20,
                alpha=0.4,
                marker=self.testmarker,
            )
            ax.legend(["train", "test"])
        ax.set_xlabel(EDFT_LABEL)
        ax.set_ylabel(EIP_LABEL)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(-0.1, 0.1)
        fig.tight_layout()
        return fig, ax

    def plot_energy_hist(self, test=False, fig_ax=None):
        fig, ax = _get_fig_ax(fig_ax, self.figsize)
        ax.hist(self.diff_DF.energy_per_atom, bins=51, color="black")
        if test:
            ax.hist(self.test_diff_DF.energy_per_atom, bins=51, color="red")
            ax.legend(["train", "test"])
        ax.set_xlabel(EIP_MINUS_EDFT_LABEL)
        ax.set_ylabel("Counts")
        # ax.set_xlim(0, 1)
        # ax.set_ylim(-0.1, 0.1)
        fig.tight_layout()
        return fig, ax

    def plot_force_scatter(self, test=False, fig_ax=None):
        fig, ax = _get_fig_ax(fig_ax, self.figsize)
        flat_dft_forces = np.concatenate(self.dft_DF.forces.array).flatten()
        flat_pot_forces = np.concatenate(self.pot_DF.forces.array).flatten()
        # dft_forces_norm = np.linalg.norm(dft_forces, axis=1)
        # pot_forces_norm = np.linalg.norm(pot_forces, axis=1)
        ax.scatter(
            flat_dft_forces,
            flat_pot_forces,
            edgecolor="black",
            facecolor="none",
            s=20,
            alpha=0.4,
        )
        if test:
            test_flat_dft_forces = np.concatenate(
                self.test_dft_DF.forces.array
            ).flatten()
            test_flat_pot_forces = np.concatenate(
                self.test_pot_DF.forces.array
            ).flatten()
            ax.scatter(
                test_flat_dft_forces,
                test_flat_pot_forces,
                edgecolor="red",
                facecolor="none",
                s=20,
                alpha=0.4,
                marker=self.testmarker,
            )
            ax.legend(["train", "test"])
        ax.set_xlabel(FDFT_LABEL)
        ax.set_ylabel(FIP_LABEL)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(-0.1, 0.1)
        fig.tight_layout()

        return fig, ax

    def plot_force_hist(self, bins=None, axis=None, test=False, fig_ax=None):
        """
        args:
        axis [None, str, int]: None or "norm" or "all" or one of [0,1,2]

        returns: fig, ax
        """
        fig, ax = _get_fig_ax(fig_ax, self.figsize)

        if axis is None:
            flatforces = np.concatenate(self.diff_DF.forces.values).flatten()
            if test:
                test_flatforces = np.concatenate(
                    self.test_diff_DF.forces.values
                ).flatten()
            ax.set_xlabel(FDIFF_LABEL)
            ax.set_xlim(-2.5, 2.5)
        elif axis == "norm":
            flatforces = np.concatenate(self.diff_DF.forces_norm.values)
            if test:
                test_flatforces = np.concatenate(self.test_diff_DF.forces_norm.values)
            ax.set_xlabel(r"||F_${DFT}$ - F_${IP}$|| [eV/$\mathrm{\AA}$]")
            ax.set_xlim(0, 2.5)
        elif axis == "all":
            flatforces = np.concatenate(self.diff_DF.forces.values)
            if test:
                test_flatforces = np.concatenate(self.test_diff_DF.forces.values)
            ax.set_xlabel(FDIFF_LABEL)
            ax.set_xlim(-2.5, 2.5)
        else:
            flatforces = np.concatenate(self.diff_DF.forces.values)[:, axis]
            if test:
                test_flatforces = np.concatenate(self.test_diff_DF.forces.values)[
                    :, axis
                ]
            ax.set_xlabel(FDIFF_LABEL)
            ax.set_xlim(-2.5, 2.5)

        ax.hist(
            flatforces,
            bins=bins,
            edgecolor="black",
            facecolor="black",
        )
        if test:
            ax.hist(
                test_flatforces,
                bins=bins,
                edgecolor="red",
                facecolor="red",
            )
            ax.legend(["train", "test"])
        ax.set_ylabel(r"Count")
        # ax.set_ylim(-0.1, 0.1)
        fig.tight_layout()

        return fig, ax


def test_load_jobs(pr, quick_test=None):
    jobs = pr.job_table()
    pot_ids = None
    if pot_ids is None:
        jobs = pr.job_table()
        jobs = jobs[jobs.hamilton.str.lower() == "atomicrex"]
        jobs = jobs[jobs.status == "finished"]
        pot_ids = jobs.id
    else:
        pot_ids = pd.Series(pot_ids)

    for ID in pot_ids:
        try:
            pot_job = pr.load(ID)
        except Exception as e:
            print(f"Failed to Load {ID}")
            print(e)
            continue

        pot = pot_job.lammps_potential
        if quick_test is not None:
            qt = quick_test(pot)
            if not qt:
                print("quick_test returned False")
            else:
                print("quick test succesfull")


def AlCuZr_db_standard_filter(row):
    calc = row.calculation
    if calc == "Dimer":
        return False
    if calc == "GrainBoundary":
        return False
    if calc == "SmallDistances":
        return False
    if row.dist_from_chull > 10.0:
        return False
    if row.min_distance > 5.0:
        return False
    if row.min_distance < 1.4:
        return False
    if row.volume / row.natoms > 150.0:
        if calc != "Surface":
            return False
    max_f = np.max(np.linalg.norm(row.data["dft_forces"], axis=1))
    if max_f > 50.0:
        return False
    return True


def SiOC_db_standard_filter(row):
    if row.max_min_distance > 4.0:
        return False
    if row.max_force > 150:
        return False
    if row.dist_from_chull > 20.0:
        return False
    if row.min_distance < 0.6:
        return False
    return True


def get_diff_DF(dft_DF, pot_DF, indices=None):
    if indices is not None:
        dft_DF = dft_DF[dft_DF.index.isin(indices)]
        pot_DF = pot_DF[pot_DF.index.isin(indices)]

    f_diff_norm = []
    f_norm_max = []
    n_atoms = []
    e_diff = pot_DF.energy - dft_DF.energy
    f_diff = pot_DF.forces - dft_DF.forces

    for f_d in f_diff:
        norm = np.linalg.norm(f_d, axis=1)
        f_diff_norm.append(norm)
        f_norm_max.append(np.max(norm))
        n_atoms.append(len(norm))

    diff_DF = pd.DataFrame(
        {
            "energy": e_diff.array,
            "forces": f_diff.array,
            "forces_norm": f_diff_norm,
            "forces_max": f_norm_max,
            "energy_per_atom": pot_DF.energy_per_atom.array
            - dft_DF.energy_per_atom.array,
            "n_atoms": n_atoms,
        },
        index=dft_DF.index,
    )
    return diff_DF


def get_standard_measures(diff_DF, ids=None):
    if ids is not None:
        ids = np.array(ids)
        diff_DF = diff_DF[diff_DF.index.isin(ids)]
        if not np.all(diff_DF.index.values == ids):
            print(
                "Warning: diff_df index != ids. Probably means that diff_DF does not contain all ids"
            )

    ermse = np.sqrt(np.mean(diff_DF.energy_per_atom**2))
    emae = diff_DF.energy_per_atom.abs().mean()
    emaxe = diff_DF.energy_per_atom.abs().max()

    # flat_force_norm = np.concatenate(diff_DF.forces_norm.values)
    # frmse = np.sqrt(np.mean(flat_force_norm**2))
    # fmae = flat_force_norm.mean()
    # fmaxe = flat_force_norm.max()

    flat_force = np.concatenate(diff_DF.forces.values).flatten()
    frmse = np.sqrt(np.mean(flat_force**2))
    fmae = np.average(np.abs(flat_force))
    fmaxe = np.max(np.abs(flat_force))

    d = {
        "nStructures": len(diff_DF),
        "nAtoms": diff_DF.n_atoms.sum(),
        "eRMSE": ermse,
        "eMAE": emae,
        "eMaxError": emaxe,
        "fRMSE": frmse,
        "fMAE": fmae,
        "fMaxError": fmaxe,
    }
    return d


def property_filter(
    db,
    ele,
    lat1,
    lat2,
    properties,
    filter_lat2=True,
    filter_a=True,
    filter_BM=True,
    BM_error=15,
    a_error=0.1,
):
    properties = properties.copy()
    eq_row = db.get(Elements=ele, lattice=lat1, calculation="Relax")
    BM = eq_row.clamped_bulk_modulus
    a = eq_row.cell[0][0]
    for complat in crystal_lattices:
        if lat1 == complat:
            continue
        properties = properties[
            properties[f"{lat1}_relaxed_energy"]
            < properties[f"{complat}_relaxed_energy"]
        ]

    if filter_lat2:
        for complat in crystal_lattices:
            if lat1 == complat or lat2 == complat:
                continue
            properties = properties[
                properties[f"{lat2}_relaxed_energy"]
                < properties[f"{complat}_relaxed_energy"]
            ]

    if filter_BM:
        properties = properties[
            properties[f"{lat1}_bulk_module"].between(BM - BM_error, BM + BM_error)
        ]
    if filter_a:
        properties = properties[
            properties[f"{lat1}_a"].between(a - a_error, a + a_error)
        ]
    return properties


def structure_to_lammps(structure):
    """
    Converts a structure to the Lammps coordinate frame

    Args:
        structure (pyiron.atomistics.structure.atoms.Atoms): Structure to convert.

    Returns:
        pyiron.atomistics.structure.atoms.Atoms: Structure with the LAMMPS coordinate frame.
    """
    prism = UnfoldingPrism(structure.cell)
    lammps_structure = structure.copy()
    lammps_structure.set_cell(prism.A)
    lammps_structure.positions = np.matmul(structure.positions, prism.R)
    return lammps_structure, prism


# This somehow leads to a completely locked db that can't be updated anymore, in a way I don't really understand
def db_lammps_prism_tag(db, potential, pr, filter_func=None, selection=None):
    lmp = setup_interactive_lammps(pr=pr, potential=potential, name="Interactive")
    for row in db.select(selection=selection, filter=filter_func, include_data=False):
        if row.get("prism", None) is not None:
            print(f"Skipped {row.id}")
            continue

        lmp.structure = ase_to_pyiron(row.toatoms())
        lmp.input.control["neigh_modify"] = "page 100000 one 10000"
        try:
            lmp.run()
            db.update(row.id, prism=False)
            print(f"Test {row.id}")
        except Exception as e:
            print("Caught exception:")
            print(e)
            print("For: ", row.id)
            db.update(row.id, prism=True)
            lmp = setup_interactive_lammps(
                pr=pr, potential=potential, name="Interactive"
            )
            print(f"Found prism {row.id}, starting again")
    return


def energy_forces_df(
    db,
    path,
    potential=None,
    pr=None,
    file=None,
    filter_func=None,
    selection=None,
    update=True,
    del_old=False,
    id_e_f_func=None,
    **filter_kwargs,
):
    filepath = handle_ef_file(file=file, path=path, del_old=del_old)
    # Change to instead filter already calculated structs, based on time or based on ids
    concat = False
    if os.path.isfile(filepath):
        df_old = pd.read_pickle(filepath, compression="gzip")
        if not update:
            print(f"{filepath} exists. Stopping.")
            return df_old

        existing_ids = df_old["db_id"]
        new_ids = []
        for row in db.select(
            selection=selection, filter=filter_func, include_data=True, **filter_kwargs
        ):
            new_ids.append(row.id)

        new_ids = pd.Series(new_ids)
        to_do_ids = new_ids[~new_ids.isin(existing_ids)]
        if len(to_do_ids) == 0:
            print("No new structures to calculate")
            return df_old

        if filter_func is None:

            def filter_func_final(row):
                if row.id in to_do_ids:
                    return True
                return False

        else:

            def filter_func_final(row):
                if row.id in to_do_ids and filter_func(row):
                    return True
                return False

        concat = True
    else:
        filter_func_final = filter_func

    if id_e_f_func is None:
        ids, energies, forces = ef_from_db_ase(
            db=db,
            pr=pr,
            potential=potential,
            selection=selection,
            filter_func=filter_func_final,
            **filter_kwargs,
        )
    else:
        ids, energies, forces = id_e_f_func()

    df = pd.DataFrame(
        {
            "db_id": ids,
            "energy": energies,
            "forces": forces,
        },
        dtype=object,
    )
    if concat and len(df) > 0:
        df = pd.concat([df_old, df])

    modify_db_df(df)
    df.to_pickle(filepath, compression="gzip", protocol=5)
    return df


def handle_ef_file(file, path, del_old):
    if file is None:
        file = "EnergyForcesDF.pickl.gz"
        file_old = "EnergyForcesDF.pick.gzip"
        filepath_old = f"{path}/{file_old}"
        if os.path.isfile(filepath_old):
            filepath = filepath_old
        else:
            filepath = f"{path}/{file}"
    else:
        filepath = f"{path}/{file}"

    if del_old:
        try:
            os.remove(filepath)
        except FileNotFoundError:
            print(f"Could not remove {filepath}, because it doesn't exist, continuing.")

    return filepath


def ef_from_db(db, pr, potential, selection, filter_func=None, **filter_kwargs):
    lmp = setup_interactive_lammps(pr=pr, potential=potential, name="InteractiveEF")
    ids = []
    energies = []
    forces = []
    for row in db.select(
        selection=selection,
        filter=filter_func,
        include_data=True,
        **filter_kwargs,
    ):
        try:
            try:
                s = ase_to_pyiron(row.toatoms())
                structure, prism = structure_to_lammps(s)
                lmp.structure = structure
                lmp.run()
            except Exception as e:
                print(f"Error occurd on {row.id}: {e}")
                print("Trying again")
                lmp = setup_interactive_lammps(
                    pr=pr, potential=potential, name="InteractiveEF"
                )
                s = ase_to_pyiron(row.toatoms())
                structure, prism = structure_to_lammps(s)
                lmp.structure = structure
                lmp.run()

            energies.append(lmp.interactive_energy_pot_getter())
            flmp = lmp.interactive_forces_getter()
            forces.append(np.matmul(flmp, prism.R.T))
            ids.append(row.id)
        except Exception as e:
            print("Failed twice for {row.id}")
            print(f"Exception: {e}")
            print("Skipping it")

    return ids, energies, forces


def setup_ase_lammps_calc(pr, potential,):
    lmp = pr.create.job.Lammps("PotWriter")
    lmp.potential = potential
    lmp.input.potential.write_file("potential.inp", cwd=pr.path)
    for file in potential.Filename[0]:
        os.system(f"cp {file} {pr.path}")

    lmpcmds = ["include potential.inp"]
    atom_types = {}
    ele_lst = potential.Species[0]
    n = 1
    for ele in ele_lst:
        atom_types[ele] = n
        n += 1
    calc = ase.calculators.lammpslib.LAMMPSlib(
        lmpcmds=lmpcmds,
        atom_types=atom_types,
    )
    return calc


def ef_from_atoms_list(atoms, pr, potential):
    cwd = os.getcwd()
    os.chdir(pr.path)
    calc = setup_ase_lammps_calc(pr, potential)
    ids = np.arange(0,len(atoms), 1,dtype=int)
    energies = []
    forces = []
    for ats in atoms:
        ats.calc = calc
        energies.append(ats.get_potential_energy())
        forces.append(ats.get_forces())
    calc.clean()
    os.chdir(cwd)
    return ids, energies, forces


def ef_from_db_ase(db, pr, potential, selection, filter_func=None, **filter_kwargs):
    cwd = os.getcwd()
    os.chdir(pr.path)
    calc = setup_ase_lammps_calc(pr, potential)
    ids = []
    energies = []
    forces = []
    for row in db.select(
        selection=selection,
        filter=filter_func,
        include_data=True,
        **filter_kwargs,
    ):
        ats = row.toatoms()
        ids.append(row.id)
        ats.calc = calc
        energies.append(ats.get_potential_energy())
        forces.append(ats.get_forces())

    calc.clean()
    os.chdir(cwd)
    return ids, energies, forces


def setup_interactive_lammps(pr, potential, name="InteractiveEF"):
    pr.remove_job(name, _unprotect=True)
    lmp = pr.create.job.Lammps(name)
    lmp.server.run_mode.interactive = True
    lmp.interactive_enforce_structure_reset = True
    lmp.potential = potential
    return lmp


def test_crystalline_potential(
    db,
    potential_pr,
    element_str,
    test_pr_path,
    properties=None,
    pot_ids=None,
    quick_test=None,
):
    current_version = 2
    # db = ase.db.connect(db_name)

    if pot_ids is None:
        jobs = potential_pr.job_table()
        jobs = jobs[jobs.hamilton.str.lower() == "atomicrex"]
        pot_ids = jobs.id
    else:
        pot_ids = pd.Series(pot_ids)

    if properties is None:
        n_pots = len(pot_ids)
        properties = init_property_df(n_pots)
        # set start to 0 for enumerate later
        start = 0
    else:
        mask = pot_ids.isin(properties.PotentialJobID)
        tested_pot_ids = pot_ids[mask]
        print(
            "pot_ids: \n",
            tested_pot_ids,
            "\nalready exist in the dataframe, ignoring them",
        )
        pot_ids = pot_ids[~mask]
        n_pots = len(pot_ids)
        # set start for enumerate later
        start = properties.index[-1] + 1
        properties2 = init_property_df(n_pots)
        properties = pd.concat([properties, properties2], ignore_index=True)

    for index, pot_job_id in enumerate(pot_ids, start=start):
        try:
            pot_job = potential_pr.load(pot_job_id)
        except:
            properties["PotentialJobID"][index] = pot_job_id
            properties["version"][index] = 201  # error when trying to load pot_job
            continue

        properties["PotentialJobID"][index] = pot_job_id
        pot = pot_job.lammps_potential
        try:
            pot = pyih.repair_pot_df(pot)
        except (ValueError, FileNotFoundError, IndexError):
            print(f"A problem occured repairing the file of potential {pot_job_id}")
            print(f"Working directory of pot job is {pot_job.working_directory}")
            properties["version"][
                index
            ] = 254  # error when trying to repair the potfile
            continue
        if quick_test is not None:
            if not quick_test(pot):
                properties["version"][
                    index
                ] = 255  # error when calculating properties or quick_test returns False
                continue

        calc_averages_and_medians(job=pot_job, df=properties, index=index)
        test_pr = Project(f"{test_pr_path}/pot_{pot_job_id}")
        try:
            calculate_crystalline_properties(
                db=db,
                properties=properties,
                index=index,
                element_str=element_str,
                test_pr=test_pr,
                pot=pot,
            )
            properties["job_path"][index] = pot_job.working_directory
            properties["version"][index] = current_version
        except Exception as e:
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"An error occured during the calculation of properties using the potential {pot_job_id}."
            )
            print("This probably there are some serious problems")
            print("Error was:")
            print(e)
            print("Continuing with the next one.")
            print(
                "--------------------------------------------------------------------------"
            )
            properties["version"][index] = 255  # error when calculating properties
            continue
    return properties


def test_crystalline_adp(
    db,
    potential_pr,
    element_str,
    test_pr_path,
    properties=None,
    pot_ids=None,
    quick_test=None,
):
    current_version = 2
    # db = ase.db.connect(db_name)

    if pot_ids is None:
        jobs = potential_pr.job_table()
        jobs = jobs[jobs.hamilton.str.lower() == "atomicrex"]
        jobs = jobs[jobs.status == "finished"]
        pot_ids = jobs.id
    else:
        pot_ids = pd.Series(pot_ids)

    if properties is None:
        n_pots = len(pot_ids)
        properties = init_property_df(n_pots)
        # set start to 0 for enumerate later
        start = 0
    else:
        mask = pot_ids.isin(properties.PotentialJobID)
        tested_pot_ids = pot_ids[mask]
        print(
            "pot_ids: \n",
            tested_pot_ids,
            "\nalready exist in the dataframe, ignoring them",
        )
        pot_ids = pot_ids[~mask]
        n_pots = len(pot_ids)
        # set start for enumerate later
        start = properties.index[-1] + 1
        properties2 = init_property_df(n_pots)
        properties = pd.concat([properties, properties2], ignore_index=True)

    for index, pot_job_id in enumerate(pot_ids, start=start):
        properties["PotentialJobID"][index] = pot_job_id
        try:
            pot_job = potential_pr.load(pot_job_id)
        except Exception as e:
            print(f"Failed to Load {pot_job_id}")
            print(e)
            print("Trying h5debug")
            try:
                pot_job = potential_pr.load(pot_job_id, convert_to_object=False)
                out = subprocess.run(
                    ["h5debug", f"{pot_job.path}.h5"],
                    stdout=subprocess.DEVNULL,
                    # stderr=subprocess.STDOUT,
                    check=True,
                    # shell=True
                )
                pot_job = potential_pr.load(pot_job_id)
            except Exception as e:
                print(f"Failed to Load {pot_job_id} after h5debug")
                print(e)
                # print(out)
                properties["PotentialJobID"][index] = pot_job_id
                properties["version"][index] = 201  # error when trying to load pot_job
                continue

        pot = pot_job.lammps_potential
        if quick_test is not None:
            if not quick_test(pot):
                print("quick_test returned False")
                properties["version"][
                    index
                ] = 255  # error when calculating properties or quick_test returns False
                continue

        # calc_averages_and_medians(job=pot_job, df=properties, index=index)
        test_pr = Project(f"{test_pr_path}/pot_{pot_job_id}")
        try:
            calculate_crystalline_properties(
                db=db,
                properties=properties,
                index=index,
                element_str=element_str,
                test_pr=test_pr,
                pot=pot,
            )
            properties["job_path"][index] = pot_job.working_directory
            properties["version"][index] = current_version
        except Exception as e:
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"An error occured during the calculation of properties using the potential {pot_job_id}."
            )
            print("This probably means there are some serious problems")
            print("Error was:")
            print(e)
            print("Continuing with the next one.")
            print(
                "--------------------------------------------------------------------------"
            )
            properties["version"][index] = 255  # error when calculating properties
            continue
    return properties


def test_crystalline_mtp(
    db,
    pot_paths,
    element_str,
    pot_elements,
    test_pr_path,
    properties=None,
    simple_test=None,
    pot_file="output.mtp",
):
    current_version = 2
    ## get hash values of paths to not test same potential twice
    path_hashes = [
        int.from_bytes(hl.blake2b(path.encode(), digest_size=8).digest(), "big")
        for path in pot_paths
    ]
    pot_ids = pd.Series(path_hashes)

    if properties is None:
        n_pots = len(pot_ids)
        properties = init_property_df(n_pots)
        # set start to 0 for enumerate later
        start = 0
    else:
        mask = pot_ids.isin(properties.PotentialJobID)
        tested_pot_ids = pot_ids[mask]
        print(
            "pot_ids: \n",
            tested_pot_ids,
            "\nalready exist in the dataframe, ignoring them",
        )
        pot_ids = pot_ids[~mask]
        n_pots = len(pot_ids)
        # set start for enumerate later
        start = properties.index[-1] + 1
        properties2 = init_property_df(n_pots)
        properties = pd.concat([properties, properties2], ignore_index=True)

    for index, (pot_job_id, path) in enumerate(zip(pot_ids, pot_paths), start=start):
        properties["PotentialJobID"][index] = pot_job_id
        properties["job_path"][index] = path
        pot = mlh.MTP_DF(
            path,
            pot_file=pot_file,
            ini_file=None,
            elements=pot_elements,
            interactive=True,
        )

        if simple_test is not None:
            if not simple_test():
                properties["version"][index] = 255  # simple test returning False
            continue

        test_pr = Project(f"{test_pr_path}/pot_{pot_job_id}")
        try:
            calculate_crystalline_properties(
                db=db,
                properties=properties,
                index=index,
                element_str=element_str,
                test_pr=test_pr,
                pot=pot,
            )
            properties["version"][index] = current_version
        except Exception as e:
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"An error occured during the calculation of properties using the potential {path}."
            )
            print("This probably there are some serious problems")
            print("Error was:")
            print(e)
            print("Continuing with the next one.")
            print(
                "--------------------------------------------------------------------------"
            )
            properties["version"][index] = 255  # error when calculating properties
            continue
    return properties


def test_crystalline_ace_potential(
    db, pot_paths, element_str, test_pr_path, properties=None, simple_test=None
):
    current_version = 2
    ## get hash values of paths to not test same potential twice
    path_hashes = [
        int.from_bytes(hl.blake2b(path.encode(), digest_size=8).digest(), "big")
        for path in pot_paths
    ]
    pot_ids = pd.Series(path_hashes)

    if properties is None:
        n_pots = len(pot_ids)
        properties = init_property_df(n_pots)
        # set start to 0 for enumerate later
        start = 0
    else:
        mask = pot_ids.isin(properties.PotentialJobID)
        tested_pot_ids = pot_ids[mask]
        print(
            "pot_ids: \n",
            tested_pot_ids,
            "\nalready exist in the dataframe, ignoring them",
        )
        pot_ids = pot_ids[~mask]
        n_pots = len(pot_ids)
        # set start for enumerate later
        start = properties.index[-1] + 1
        properties2 = init_property_df(n_pots)
        properties = pd.concat([properties, properties2], ignore_index=True)

    for index, (pot_job_id, path) in enumerate(zip(pot_ids, pot_paths), start=start):
        properties["PotentialJobID"][index] = pot_job_id
        properties["job_path"][index] = path
        pot = pyih.PACE_DF(path, elements=[f"{element_str}"])

        if simple_test is not None:
            if not simple_test():
                properties["version"][index] = 255  # simple test returning False
            continue

        test_pr = Project(f"{test_pr_path}/pot_{pot_job_id}")
        try:
            calculate_crystalline_properties(
                db=db,
                properties=properties,
                index=index,
                element_str=element_str,
                test_pr=test_pr,
                pot=pot,
            )
            properties["version"][index] = current_version
        except Exception as e:
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"An error occured during the calculation of properties using the potential {path}."
            )
            print("This probably there are some serious problems")
            print("Error was:")
            print(e)
            print("Continuing with the next one.")
            print(
                "--------------------------------------------------------------------------"
            )
            properties["version"][index] = 255  # error when calculating properties
            continue
    return properties


def test_crystalline_nnp_potential(
    db, pot_paths, element_str, test_pr_path, cutoff, properties=None, simple_test=None
):
    current_version = 2
    ## get hash values of paths to not test same potential twice
    path_hashes = [
        int.from_bytes(hl.blake2b(path.encode(), digest_size=8).digest(), "big")
        for path in pot_paths
    ]
    pot_ids = pd.Series(path_hashes)

    if properties is None:
        n_pots = len(pot_ids)
        properties = init_property_df(n_pots)
        # set start to 0 for enumerate later
        start = 0
    else:
        mask = pot_ids.isin(properties.PotentialJobID)
        tested_pot_ids = pot_ids[mask]
        print(
            "pot_ids: \n",
            tested_pot_ids,
            "\nalready exist in the dataframe, ignoring them",
        )
        pot_ids = pot_ids[~mask]
        n_pots = len(pot_ids)
        # set start for enumerate later
        start = properties.index[-1] + 1
        properties2 = init_property_df(n_pots)
        properties = pd.concat([properties, properties2], ignore_index=True)

    for index, (pot_job_id, path) in enumerate(zip(pot_ids, pot_paths), start=start):
        properties["PotentialJobID"][index] = pot_job_id
        properties["job_path"][index] = path
        pot = pyih.NNP_DF(
            path=path,
            elements=[element_str],
            cutoff=cutoff,
        )

        if simple_test is not None:
            if not simple_test():
                properties["version"][index] = 255  # simple test returning False
            continue

        test_pr = Project(f"{test_pr_path}/pot_{pot_job_id}")
        try:
            calculate_crystalline_properties(
                db=db,
                properties=properties,
                index=index,
                element_str=element_str,
                test_pr=test_pr,
                pot=pot,
            )
            properties["version"][index] = current_version
        except Exception as e:
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"An error occured during the calculation of properties using the potential {path}."
            )
            print("This probably there are some serious problems")
            print("Error was:")
            print(e)
            print("Continuing with the next one.")
            print(
                "--------------------------------------------------------------------------"
            )
            properties["version"][index] = 255  # error when calculating properties
            continue
    return properties


def test_pot_str_potential(
    db, pot_strs, element_str, test_pr_path, properties=None, simple_test=None
):
    current_version = 2
    ## get hash values of paths to not test same potential twice
    pot_hashes = [
        int.from_bytes(hl.blake2b(path.encode(), digest_size=8).digest(), "big")
        for path in pot_strs
    ]
    pot_ids = pd.Series(pot_hashes)

    if properties is None:
        n_pots = len(pot_ids)
        properties = init_property_df(n_pots)
        # set start to 0 for enumerate later
        start = 0
    else:
        mask = pot_ids.isin(properties.PotentialJobID)
        tested_pot_ids = pot_ids[mask]
        print(
            "pot_ids: \n",
            tested_pot_ids,
            "\nalready exist in the dataframe, ignoring them",
        )
        pot_ids = pot_ids[~mask]
        n_pots = len(pot_ids)
        # set start for enumerate later
        start = properties.index[-1] + 1
        properties2 = init_property_df(n_pots)
        properties = pd.concat([properties, properties2], ignore_index=True)

    for index, (pot_job_id, pot) in enumerate(zip(pot_ids, pot_strs), start=start):
        properties["PotentialJobID"][index] = pot_job_id
        properties["job_path"][index] = pot

        if simple_test is not None:
            if not simple_test():
                properties["version"][index] = 255  # simple test returning False
            continue

        test_pr = Project(f"{test_pr_path}/pot_{pot_job_id}")
        try:
            calculate_crystalline_properties(
                db=db,
                properties=properties,
                index=index,
                element_str=element_str,
                test_pr=test_pr,
                pot=pot,
            )
            properties["version"][index] = current_version
        except Exception as e:
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"An error occured during the calculation of properties using the potential {pot}."
            )
            print("This probably there are some serious problems")
            print("Error was:")
            print(e)
            print("Continuing with the next one.")
            print(
                "--------------------------------------------------------------------------"
            )
            properties["version"][index] = 255  # error when calculating properties
            continue
    return properties


def calculate_crystalline_properties(db, properties, index, element_str, test_pr, pot):
    # version 1
    interactive = True
    for structure_str in crystal_lattices:
        row = db.get(Elements=element_str, lattice=structure_str, calculation="Relax")
        print(f"Calculating {structure_str} properties")
        structure = ase_to_pyiron(row.toatoms())
        try:
            relax = relax_or_load_job(test_pr, f"pid_{row.pyiron_id}", structure, pot)
        except Exception as e:
            print("Problem calculating or loading relax. Error was:")
            print(e)
            print("Continuing with next structure")
            continue

        if relax.status != "finished":
            print("relax didn't finish, continuing with next structure")
            continue

        # Only run if relax finished
        relax_structure = relax.get_structure()
        print("Structure relaxed succesfully")
        properties[f"{structure_str}_a"][index] = np.linalg.norm(
            relax["output/generic/cells"][-1][0]
        )
        relaxed_energy = relax["output/generic/energy_pot"][-1] / len(structure)
        properties[f"{structure_str}_relaxed_energy"][index] = relaxed_energy

        if structure_str == "hcp":
            properties["hcp_c"][index] = np.linalg.norm(
                relax["output/generic/cells"][-1][2]
            )
            interactive = False

        try:
            murn = murnaghan_or_load_jobpath(
                test_pr,
                f"pid_{row.pyiron_id}_murn",
                relax_structure,
                pot,
                interactive=interactive,
            )
        except Exception as e:
            print("Problem calculating murn. Error was:")
            print(e)
        try:
            elastic = elastic_or_load_jobpath(
                test_pr,
                f"pid_{row.pyiron_id}_elastic",
                relax_structure,
                pot,
                interactive=interactive,
            )
        except Exception as e:
            print("Problem calculating elastic. Error was:")
            print(e)

        if murn.status == "finished":
            print("Succesfully ran Murnaghan job")
            properties[f"{structure_str}_bulk_module"][index] = get_bulk_module(murn)

        if elastic.status == "finished":
            print("Succesfully ran ElasticTensor job")
            elastic_tensor = get_elastic_tensor(elastic)
            properties[f"{structure_str}_c11"][index] = elastic_tensor[0, 0]
            properties[f"{structure_str}_c12"][index] = elastic_tensor[0, 1]
            properties[f"{structure_str}_c44"][index] = elastic_tensor[3, 3]
            if structure_str == "hcp":
                properties[f"{structure_str}_c13"][index] = elastic_tensor[0, 2]
                properties[f"{structure_str}_c33"][index] = elastic_tensor[2, 2]
                properties[f"{structure_str}_c66"][index] = elastic_tensor[5, 5]

        ## properties that need to reference the relaxed energy
        for s_row in db.select(
            Elements=element_str, lattice=structure_str, calculation="Surface"
        ):
            try:
                structure = ase_to_pyiron(s_row.toatoms())
                hkl_str = s_row.path.split("/")[9]
                name = f"pid_{row.pyiron_id}_{structure_str}{hkl_str}"
                surface = surface_or_load_jobpath(test_pr, name, structure, pot)
                area = pyih.surface_areas(structure)[0]
                surface_energy = (
                    surface["output/generic/energy_pot"][-1]
                    - len(structure) * relaxed_energy
                ) / area
                properties[f"{structure_str}{hkl_str}_surface_energy"][
                    index
                ] = surface_energy
            except Exception as e:
                print(f"Problem calculating {structure_str}{hkl_str}. Error was:")
                print(e)

        try:
            vac = unrelaxed_vac_or_load_jobpath(
                test_pr, f"unrelaxed_{structure_str}_vacancy", relax_structure, pot
            )
            natoms = len(vac["input/structure/positions"])
            energy = vac["output/generic/energy_pot"][-1]

            vacancy_energy = calc_vacancy_energy(
                energy=energy, natoms=natoms, bulk_energy=relaxed_energy
            )
            properties[f"{structure_str}_vacancy_energy"][index] = vacancy_energy
        except Exception as e:
            print("Problem calculating unrelaxed vacancy. Error was:")
            print(e)

    # temporarily deactivate these because they take too long
    # Probably better to run seperately only for at least somewhat ok potentials
    # properties["standard_md"][index] = standardMD_or_load_jobpath(test_pr, element=element_str, potential=pot)
    # properties["extreme_md"][index] = extremeMD_or_load_jobpath(test_pr, element=element_str, potential=pot)


def init_property_df(n_pots):
    property_dict = {
        # added in version 0
        "PotentialJobID": np.full(n_pots, 0, dtype=np.uint),
        "fcc_a": np.full(n_pots, np.nan),
        "bcc_a": np.full(n_pots, np.nan),
        "sc_a": np.full(n_pots, np.nan),
        "diamond_a": np.full(n_pots, np.nan),
        "hcp_a": np.full(n_pots, np.nan),
        "hcp_c": np.full(n_pots, np.nan),
        "fcc_bulk_module": np.full(n_pots, np.nan),
        "bcc_bulk_module": np.full(n_pots, np.nan),
        "sc_bulk_module": np.full(n_pots, np.nan),
        "diamond_bulk_module": np.full(n_pots, np.nan),
        "hcp_bulk_module": np.full(n_pots, np.nan),
        "fcc_relaxed_energy": np.full(n_pots, np.nan),
        "bcc_relaxed_energy": np.full(n_pots, np.nan),
        "sc_relaxed_energy": np.full(n_pots, np.nan),
        "diamond_relaxed_energy": np.full(n_pots, np.nan),
        "hcp_relaxed_energy": np.full(n_pots, np.nan),
        "fcc111_surface_energy": np.full(n_pots, np.nan),
        "fcc110_surface_energy": np.full(n_pots, np.nan),
        "fcc100_surface_energy": np.full(n_pots, np.nan),
        "bcc111_surface_energy": np.full(n_pots, np.nan),  # new surface, not for Cu/Zr
        "bcc110_surface_energy": np.full(n_pots, np.nan),
        "bcc100_surface_energy": np.full(n_pots, np.nan),
        "hcp0001_surface_energy": np.full(n_pots, np.nan),
        "hcp10m10_surface_energy": np.full(n_pots, np.nan),
        "diamond100_surface_energy": np.full(n_pots, np.nan),
        "diamond111_surface_energy": np.full(n_pots, np.nan),
        "fcc_vacancy_energy": np.full(n_pots, np.nan),
        "bcc_vacancy_energy": np.full(n_pots, np.nan),
        "sc_vacancy_energy": np.full(n_pots, np.nan),
        "diamond_vacancy_energy": np.full(n_pots, np.nan),
        "hcp_vacancy_energy": np.full(n_pots, np.nan),
        "fcc_rattle_energies": np.full(n_pots, np.nan),
        "bcc_rattle_energies": np.full(n_pots, np.nan),
        "sc_rattle_energies": np.full(n_pots, np.nan),
        "diamond_rattle_energies": np.full(n_pots, np.nan),
        "hcp_rattle_energies": np.full(n_pots, np.nan),
        "hcp_vacancy_forces": np.full(n_pots, np.nan),
        "fcc_rattle_forces": np.full(n_pots, np.nan),
        "bcc_rattle_forces": np.full(n_pots, np.nan),
        "sc_rattle_forces": np.full(n_pots, np.nan),
        "diamond_rattle_forces": np.full(n_pots, np.nan),
        "hcp_rattle_forces": np.full(n_pots, np.nan),
        # average and median quantities for all structures:
        "avg_force_err": np.full(n_pots, np.nan),
        "median_force_err": np.full(n_pots, np.nan),
        "max_force_err": np.full(n_pots, np.nan),
        "avg_en_err": np.full(n_pots, np.nan),
        "median_en_err": np.full(n_pots, np.nan),
        "max_en_err": np.full(n_pots, np.nan),
        # added in version 1
        # Check if md runs work or cause any problems
        "standard_md": np.full(n_pots, False, bool),
        "extreme_md": np.full(n_pots, False, bool),
        # Add a version number to be able to remember which property I added at which point (added in version 1)
        # Also use to remeber if calculating properties failed by setting to a number >200 and <256 if an error occurs
        "version": np.full(n_pots, 0, dtype=np.uint8),
    }
    # store relevant values of calculated elastic tensors. added in version 1
    for s in ["fcc", "bcc", "sc", "diamond", "hcp"]:
        property_dict[f"{s}_c11"] = np.full(n_pots, np.nan)
        property_dict[f"{s}_c12"] = np.full(n_pots, np.nan)
        property_dict[f"{s}_c44"] = np.full(n_pots, np.nan)

    property_dict["hcp_c13"] = np.full(n_pots, np.nan)
    property_dict["hcp_c33"] = np.full(n_pots, np.nan)
    property_dict["hcp_c66"] = np.full(n_pots, np.nan)

    # verion 2
    property_dict["job_path"] = np.full(n_pots, "not set", dtype=object)

    # return property_dict
    return pd.DataFrame(property_dict)


def calculate_mixed_properties_ARPot(
    db,
    potential_pr,
    ele1,
    ele2,
    test_pr_path,
    properties=None,
    pot_ids=None,
    quick_test=None,
):
    current_version = 2
    # db = ase.db.connect(db_name)

    if pot_ids is None:
        jobs = potential_pr.job_table()
        jobs = jobs[jobs.hamilton.str.lower() == "atomicrex"]
        pot_ids = jobs.id
    else:
        pot_ids = pd.Series(pot_ids)

    if properties is None:
        n_pots = len(pot_ids)
        properties = init_mixedCrystal_df(n_pots, db, f"{ele1}{ele2}", ele2)
        # set start to 0 for enumerate later
        start = 0
    else:
        mask = pot_ids.isin(properties.PotentialJobID)
        tested_pot_ids = pot_ids[mask]
        print(
            "pot_ids: \n",
            tested_pot_ids,
            "\nalready exist in the dataframe, ignoring them",
        )
        pot_ids = pot_ids[~mask]
        n_pots = len(pot_ids)
        # set start for enumerate later
        start = properties.index[-1] + 1
        properties2 = init_mixedCrystal_df(n_pots, db, f"{ele1}{ele2}", ele2)
        properties = pd.concat([properties, properties2], ignore_index=True)

    for index, pot_job_id in enumerate(pot_ids, start=start):
        try:
            pot_job = potential_pr.load(pot_job_id)
        except:
            properties["PotentialJobID"][index] = pot_job_id
            properties["version"][index] = 201  # error when trying to load pot_job
            continue

        properties["PotentialJobID"][index] = pot_job_id
        properties["job_path"][index] = pot_job.working_directory
        pot = pot_job.lammps_potential
        try:
            pot = pyih.repair_pot_df(pot)
        except (ValueError, FileNotFoundError, IndexError):
            print(f"A problem occured repairing the file of potential {pot_job_id}")
            print(f"Working directory of pot job is {pot_job.working_directory}")
            properties["version"][
                index
            ] = 254  # error when trying to repair the potfile
            continue
        if quick_test is not None:
            if not quick_test(pot):
                properties["version"][
                    index
                ] = 255  # error when calculating properties or quick_test returns False
                continue

        pot_job_id = pot_job.id
        current_version = 2
        test_pr = Project(f"{test_pr_path}/pot_{pot_job_id}")
        try:
            calculate_mixed_relaxed_energies(
                db=db,
                ele1=ele1,
                ele2=ele2,
                pot=pot,
                pr=test_pr,
                df=properties,
                index=index,
            )
            properties["version"][index] = current_version
        except Exception as e:
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"An error occured during the calculation of properties using the potential {pot_job_id}."
            )
            print("This probably there are some serious problems")
            print("Error was:")
            print(e)
            print("Continuing with the next one.")
            print(
                "--------------------------------------------------------------------------"
            )
            properties["version"][index] = 255  # error when calculating properties
            continue
    return properties


def filter_by_path_hashes(paths, properties, init_df_func, **init_df_kwargs):
    path_hashes = [
        int.from_bytes(hl.blake2b(path.encode(), digest_size=8).digest(), "big")
        for path in paths
    ]
    pot_ids = pd.Series(path_hashes)
    if properties is None:
        n_pots = len(pot_ids)
        properties = init_df_func(n_pots, **init_df_kwargs)
        # set start to 0 for enumerate later
        start = 0
    else:
        mask = pot_ids.isin(properties.PotentialJobID)
        tested_pot_ids = pot_ids[mask]
        print(
            "pot_ids: \n",
            tested_pot_ids,
            "\nalready exist in the dataframe, ignoring them",
        )
        pot_ids = pot_ids[~mask]
        n_pots = len(pot_ids)
        # set start for enumerate later
        start = properties.index[-1] + 1
        properties2 = init_df_func(n_pots, **init_df_kwargs)
        properties = pd.concat([properties, properties2], ignore_index=True)
    return start, pot_ids, properties


def property_calc_error_msg(id, e):
    print("--------------------------------------------------------------------------")
    print(
        f"An error occured during the calculation of properties using the potential {id}."
    )
    print("This probably there are some serious problems")
    print("Error was:")
    print(e)
    print("Continuing with the next one.")
    print("--------------------------------------------------------------------------")


def calculate_mixed_properties_POT(
    db,
    path,
    pot,
    ele1,
    ele2,
    test_pr_path,
    properties=None,
    quick_test=None,
    loc_filename=None,
    del_old=False,
):
    series = None
    if loc_filename is not None:
        filepath = f"{path}/{loc_filename}"
        if os.path.isfile(filepath):
            if del_old:
                os.remove(filepath)
            else:
                series = pd.read_pickle(filepath)

    current_version = 2
    # db = ase.db.connect(db_name)
    start, pot_ids, properties = filter_by_path_hashes(
        paths=[path],
        properties=properties,
        init_df_func=init_mixedCrystal_df,
        db=db,
        mixed_ele_str=f"{ele1}{ele2}",
        fraction_ele_str=ele2,
    )

    for index, pot_job_id in enumerate(pot_ids, start=start):
        if series is not None:
            if len(properties.columns) == len(series.index):
                if not np.all(properties.columns != series.index):
                    print(
                        "Warning: Length of read series and constructure properties dataframe match,"
                        "but they have different columns. Setting missing values in series to nan."
                    )
                series = series.reindex(properties.columns, fill_value=np.nan)
            else:
                print(
                    "Warning: Length of read series and constructure properties dataframe don't match, reindexing series"
                )
                series = series.reindex(properties.columns, fill_value=np.nan)
            series.name = index
            properties.iloc[index] = series
            continue

        properties["PotentialJobID"][index] = pot_job_id
        properties["job_path"][index] = path
        if quick_test is not None:
            if not quick_test(pot):
                properties["version"][
                    index
                ] = 255  # error when calculating properties or quick_test returns False
                continue

        current_version = 2
        test_pr = Project(f"{test_pr_path}/pot_{pot_job_id}")
        try:
            calculate_mixed_relaxed_energies(
                db=db,
                ele_lst=[ele1, ele2],
                pot=pot,
                pr=test_pr,
                df=properties,
                index=index,
            )
            properties["version"][index] = current_version
        except Exception as e:
            property_calc_error_msg(id=pot_job_id, e=e)
            properties["version"][index] = 255  # error when calculating properties
            continue

        if loc_filename is not None:
            series = properties.iloc[index]
            series.to_pickle(filepath)

    return properties


def calculate_mixed_relaxed_energies(db, ele_lst, pot, pr, df, index):
    mixed_ele_str = ""
    energy_dict = {}
    for ele in ele_lst:
        lmp = setup_interactive_lammps(pr=pr, potential=pot, name='GenericInteractiveLmp')
        lmp.calc_minimize(pressure=0, max_iter=100)
        s = pr.create.structure.ase.bulk(ele)
        lmp.structure = s
        lmp.run()
        numele = s.numbers[0]
        e_ele = lmp.interactive_energy_pot_getter() / len(s)
        energy_dict[numele] = e_ele
        mixed_ele_str += ele

    for row in db.select(Elements=mixed_ele_str, Crystal=True, calculation="Relax"):
        lmp = setup_interactive_lammps(pr=pr, potential=pot, name='GenericInteractiveLmp')
        s_str = row.path.split("/")[-4]
        s = ase_to_pyiron(row.toatoms())
        lmp.calc_minimize(pressure=0, max_iter=100)
        structure, prism = structure_to_lammps(s)
        lmp.structure = structure
        lmp.run()
        energy = lmp.output.energy_pot[-1]
        e = energy / row.natoms
        df[f"{s_str}_relaxed_energy"][index] = e

        mixing_energy = dbh.get_formation_enthalpy(
            energy=energy,
            numbers=row.numbers,
            energy_dict=energy_dict,
        )
        df[f"{s_str}_mixing_energy"][index] = mixing_energy


def init_mixedCrystal_df(n_pots, db, mixed_ele_str, fraction_ele_str):
    property_dict = {
        # added in version 0
        "PotentialJobID": np.full(n_pots, 0, dtype=np.uint),
        # Add a version number to be able to remember which property I added at which point (added in version 1)
        # Also use to remeber if calculating properties failed by setting to a number >200 and <256 if an error occurs
        "version": np.full(n_pots, 0, dtype=np.uint8),
    }
    for row in db.select(Elements=mixed_ele_str, calculation="Relax", Crystal=True):
        s_str = row.path.split("/")[-4]
        property_dict[f"{s_str}_relaxed_energy"] = np.full(n_pots, np.nan)
        property_dict[f"{s_str}_mixing_energy"] = np.full(n_pots, np.nan)
        symbols = np.array(row.symbols)
        nEle = len(symbols[symbols == fraction_ele_str])
        property_dict[f"{s_str}_fraction{fraction_ele_str}"] = np.full(
            n_pots, nEle / row.natoms
        )
        property_dict[f"{s_str}_melting_point"] = np.full(n_pots, np.nan)
        property_dict[f"{s_str}_calphy_path"] = np.full(n_pots, "not set", dtype=object)
    # verion 2
    property_dict["job_path"] = np.full(n_pots, "not set", dtype=object)
    # return property_dict
    return pd.DataFrame(property_dict)


def calc_averages_and_medians(job, df, index):
    try:
        f_forces = job["structures/fit_properties/atomic-forces/final_value"]
        t_forces = job["structures/fit_properties/atomic-forces/target_value"]
        delta_f = f_forces - t_forces
        l2_delta_f_abs = np.linalg.norm(delta_f, axis=1)

        df["avg_force_err"][index] = np.nanmean(l2_delta_f_abs)
        df["median_force_err"][index] = np.nanmedian(l2_delta_f_abs)
        df["max_force_err"][index] = np.nanmax(l2_delta_f_abs)

        f_energy = job["structures/fit_properties/atomic-energy/final_value"]
        t_energy = job["structures/fit_properties/atomic-energy/target_value"]
        delta_e = np.abs(f_energy - t_energy)
        df["avg_en_err"][index] = np.mean(delta_e)
        df["median_en_err"][index] = np.median(delta_e)
        df["max_en_err"][index] = np.max(delta_e)
    except Exception as e:
        print("Could not calculate average force and energy errors")
        print("Error was:")
        print(e)


def calc_surf_energy(energy, structure, bulk_energy):
    area = pyih.surface_areas(structure)[0]
    return (energy - len(structure) * bulk_energy) / area


def calc_vacancy_energy(energy, natoms, bulk_energy):
    return energy - natoms * bulk_energy


def relax_or_load_job(project, name, structure, potential):
    job = project.load(name)
    if job is None:
        job = project.create_job("Lammps", name)
        job.structure = structure
        job.potential = potential
        job.calc_minimize(pressure=np.zeros(3), max_iter=1000)
        job.run()
    return job


def unrelaxed_vac_or_load_jobpath(project, name, structure, potential):
    job = project.load(name, convert_to_object=False)
    if job is None:
        structure = structure.repeat((5, 5, 5))
        del structure[0]
        job = project.create_job("Lammps", name)
        job.structure = structure
        job.potential = potential
        job.run()
    return job


def standardMD_or_load_jobpath(project, element, potential):
    name = "standard_md"
    md = project.load(name, convert_to_object=False)
    if md is None:
        md = project.create_job("Lammps", name)
        md.structure = project.create_ase_bulk(element).repeat(15)
        md.potential = potential
        md.calc_md(temperature=273, pressure=0.0001, n_ionic_steps=10000, n_print=5000)
        md.run()
    if md.status == "finished":
        return True
    else:
        return False


def extremeMD_or_load_jobpath(project, element, potential):
    name = "extreme_md"
    md = project.load(name, convert_to_object=False)
    if md is None:
        md = project.create_job("Lammps", name)
        md.structure = project.create_ase_bulk(element).repeat(15)
        md.potential = potential
        md.calc_md(temperature=3000, pressure=1, n_ionic_steps=10000, n_print=5000)
        md.run()
    if md.status == "finished":
        return True
    else:
        return False


def murnaghan_or_load_jobpath(project, name, structure, potential, interactive=True):
    murn = project.load(name, convert_to_object=False)
    if murn is None:
        murn = project.create_job("Murnaghan", name)
        ref = project.create_job("Lammps", f"{name}_ref")
        murn.ref_job = ref
        murn.ref_job.structure = structure
        murn.ref_job.potential = potential
        ref.potential = potential
        if interactive:
            murn.ref_job.server.run_mode.interactive = True
        murn.input["num_points"] = 7
        murn.input["vol_range"] = 0.01
        murn.run()
    return murn


def elastic_or_load_jobpath(project, name, structure, potential, interactive=True):
    elastic = project.load(name, convert_to_object=False)
    if elastic is None:
        elastic = project.create_job("ElasticTensor", name)
        ref = project.create_job("Lammps", f"{name}_ref")
        elastic.ref_job = ref
        elastic.ref_job.structure = structure
        elastic.ref_job.potential = potential
        if interactive:
            elastic.ref_job.server.run_mode.interactive = True
        elastic.input["use_pressure"] = False
        elastic.run()
    return elastic


def surface_or_load_jobpath(project, name, structure, potential):
    surface = project.load(name, convert_to_object=False)
    if surface is None:
        surface = project.create_job("Lammps", name)
        surface.structure = structure
        surface.potential = potential
        surface.calc_minimize(max_iter=100)
        surface.run()
    return surface


def get_bulk_module(murnaghan_job):
    try:
        return murnaghan_job["output/equilibrium_bulk_modulus"]
    except:
        return np.nan


def get_elastic_tensor(elastic_job):
    try:
        return elastic_job["output/elastic_tensor"]
    except:
        return np.full((6, 6), np.nan)


def plot_Pareto(x_prop, y_prop, df, ref_df, x_rel_err=0.3, y_rel_err=0.3, save=False):
    x = abs(df[x_prop])
    y = abs(df[y_prop])
    x_max = abs(x_rel_err * ref_df[x_prop][0])
    y_max = abs(y_rel_err * ref_df[y_prop][0])
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel(f"{x_prop} error")
    ax.set_ylabel(f"{y_prop} error")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    if save:
        fig.savefig(
            f"/nfshome/leimeroth/Presentations/Figures/ParetoPlots/{x_prop}_{y_prop}.jpeg",
            dpi=300,
        )
    return fig, ax


def get_mixed_crystal_structure_names(db, ele1, ele2):
    structure_names = []
    mixed_str = f"{ele1}{ele2}"
    for row in db.select(Elements=mixed_str, Crystal=True, calculation="Relax"):
        s_name = row.path.split("/")[-4]
        structure_names.append(s_name)
    return structure_names


def get_mixed_crystal_dft_properties(db, ele1, ele2):
    mixed_ele_str = f"{ele1}{ele2}"
    df = init_mixedCrystal_df(1, db, mixed_ele_str=mixed_ele_str, fraction_ele_str=ele2)
    lat1 = elements_lattices[ele1]
    ele1_row = db.get(Elements=ele1, lattice=lat1, calculation="Relax")
    e_ele1 = ele1_row.nist_normed_energy / ele1_row.natoms

    lat2 = elements_lattices[ele2]
    ele2_row = db.get(Elements=ele2, lattice=lat2, calculation="Relax")
    e_ele2 = ele2_row.nist_normed_energy / ele2_row.natoms

    for row in db.select(Elements=mixed_ele_str, Crystal=True, calculation="Relax"):
        s_str = row.path.split("/")[-4]
        symbols = np.array(row.symbols)
        e = row.nist_normed_energy / row.natoms
        df[f"{s_str}_relaxed_energy"][0] = e
        n_ele1 = len(symbols[symbols == ele1])
        n_ele2 = len(symbols[symbols == ele2])
        eq_energy = (n_ele1 * e_ele1 + n_ele2 * e_ele2) / row.natoms
        mixing_energy = e - eq_energy
        df[f"{s_str}_mixing_energy"][0] = mixing_energy

    return df


def get_crystalline_dft_properties(db, element_str):
    dft_properties = init_property_df(1)
    for lat in crystal_lattices:
        row = db.get(Elements=element_str, lattice=lat, calculation="Relax")
        ref_energy = row.nist_normed_energy / row.natoms
        dft_properties[f"{lat}_a"][0] = np.linalg.norm(row.cell[0])
        dft_properties[f"{lat}_relaxed_energy"][0] = ref_energy
        dft_properties[f"{lat}_bulk_module"][0] = row.get(
            "clamped_bulk_modulus", default=np.nan
        )

        elastic_tensor = row.data["ElasticTensor"]
        dft_properties[f"{lat}_c11"][0] = elastic_tensor[0, 0]
        dft_properties[f"{lat}_c12"][0] = elastic_tensor[0, 1]
        dft_properties[f"{lat}_c44"][0] = elastic_tensor[3, 3]
        if lat == "hcp":
            dft_properties[f"{lat}_c13"][0] = elastic_tensor[0, 2]
            dft_properties[f"{lat}_c33"][0] = elastic_tensor[2, 2]
            dft_properties[f"{lat}_c66"][0] = elastic_tensor[5, 5]
            dft_properties["hcp_c"][0] = np.linalg.norm(row.cell[2])

        for row in db.select(Elements=element_str, lattice=lat, calculation="Surface"):
            hkl_str = row.path.split("/")[9]
            structure = row.toatoms()
            area = pyih.surface_areas(structure)[0]
            surface_energy = (row.nist_normed_energy - row.natoms * ref_energy) / area
            dft_properties[f"{lat}{hkl_str}_surface_energy"][0] = surface_energy

        # Calc vacancy energy
        try:
            row = db.get(Elements=element_str, lattice=lat, calculation="Vacancy")
            dft_properties[f"{lat}_vacancy_energy"][0] = (
                row.nist_normed_energy - row.natoms * ref_energy
            )
        except:
            print(f"Problem with {element_str} {lat} vacancy energy")

    return dft_properties


def get_dft_df(
    db,
    energy_key="nist_normed_energy",
    forces_key="dft_forces",
    df_file=None,
    selection=None,
    filter_func=None,
    del_old=False,
    **filter_kwargs,
):
    if df_file is None:
        df_file = "DFT_EF.pckl.gz"
    if del_old:
        os.remove(df_file)

    if os.path.isfile(df_file):
        return pd.read_pickle(df_file)

    energies = []
    forces = []
    ids = []
    atoms_lst = []
    chull_dist = []
    for row in db.select(
        selection=selection, filter=filter_func, include_data=True, **filter_kwargs
    ):
        atoms_lst.append(row.toatoms())
        energies.append(row[energy_key])
        forces.append(row.data[forces_key])
        ids.append(row.id)
        chull_dist.append(row.dist_from_chull)

    df = pd.DataFrame(
        {
            "db_id": ids,
            "ase_atoms": atoms_lst,
            "energy": energies,
            "forces": forces,
            "chull_dist": chull_dist,
        },
        dtype=object,
    )
    modify_db_df(df)
    # pah.compute_convexhull_dist(df)
    df.to_pickle(df_file)
    return df


def sort_df_by_db_id(df):
    df.sort_values("db_id", inplace=True)


def filter_df(df, db_series=None):
    if db_series is not None:
        df.drop(df[~df.db_id.isin(db_series)].index, inplace=True)


def add_n_atoms_column(df):
    df["n_atoms"] = df.forces.apply(len)


def add_epa_column(df):
    df["energy_per_atom"] = df["energy"] / df["n_atoms"]


def modify_db_df(df):
    sort_df_by_db_id(df)
    filter_df(df)
    add_n_atoms_column(df)
    add_epa_column(df)
    df.set_index("db_id", inplace=True, verify_integrity=True)


def quick_test_Cu(pot):
    pr = Project("/scratch/leimeroth/scratch/pyiron/projects/tmp")
    pr.remove_job("Interactive", _unprotect=True)
    try:
        lmp = pr.create.job.Lammps("Interactive")
        lmp.potential = pot
        lmp.structure = pr.create.structure.ase.bulk("Cu", "fcc", a=3.634, cubic=True)
        lmp.server.run_mode.interactive = True
        lmp.interactive_enforce_structure_reset = True
        lmp.calc_minimize(pressure=np.zeros(3), max_iter=20)
        lmp.run()
        e_fcc = lmp.interactive_energy_pot_getter() / 4

        lmp.structure = pr.create.structure.ase.bulk("Cu", "hcp", a=2.564, c=4.222)
        lmp.run()
        e_hcp = lmp.interactive_energy_pot_getter() / 2
        if e_hcp < e_fcc:
            return False

        lmp.structure = pr.create.structure.ase.bulk("Cu", "sc", a=2.41, cubic=True)
        lmp.run()
        e_sc = lmp.interactive_energy_pot_getter()
        if e_sc < e_fcc:
            return False

        lmp.structure = pr.create.structure.ase.bulk("Cu", "bcc", a=2.89, cubic=True)
        lmp.run()
        e_bcc = lmp.interactive_energy_pot_getter() / 2
        if e_bcc < e_fcc:
            return False

        lmp.structure = pr.create.structure.ase.bulk(
            "Cu", "diamond", a=5.36, cubic=True
        )
        lmp.run()
        e_dia = lmp.interactive_energy_pot_getter() / 8
        if e_dia < e_fcc:
            return False

    except Exception as e:
        print("Error in interactive job")
        print(e)
        return False

    return True


def quick_test_Al(pot):
    pr = Project("/scratch/leimeroth/scratch/pyiron/projects/tmp")
    pr.remove_job("Interactive", _unprotect=True)
    try:
        lmp = pr.create.job.Lammps("Interactive")
        lmp.potential = pot
        lmp.structure = pr.create.structure.ase.bulk("Al", "fcc", a=4.0407, cubic=True)
        lmp.server.run_mode.interactive = True
        lmp.interactive_enforce_structure_reset = True
        lmp.calc_minimize(pressure=np.zeros(3), max_iter=20)
        lmp.run()
        e_fcc = lmp.interactive_energy_pot_getter() / 4

        lmp.structure = pr.create.structure.ase.bulk("Al", "hcp", a=2.864, c=4.681)
        lmp.run()
        e_hcp = lmp.interactive_energy_pot_getter() / 2
        if e_hcp < e_fcc:
            return False

        lmp.structure = pr.create.structure.ase.bulk("Al", "bcc", a=3.233, cubic=True)
        lmp.run()
        e_bcc = lmp.interactive_energy_pot_getter() / 2
        if e_bcc < e_fcc:
            return False

        lmp.structure = pr.create.structure.ase.bulk("Al", "sc", a=2.722, cubic=True)
        lmp.run()
        e_sc = lmp.interactive_energy_pot_getter()
        if e_sc < e_fcc:
            return False

        lmp.structure = pr.create.structure.ase.bulk(
            "Al", "diamond", a=6.044, cubic=True
        )
        lmp.run()
        e_dia = lmp.interactive_energy_pot_getter() / 8
        if e_dia < e_fcc:
            return False

    except Exception as e:
        print("Error in interactive job")
        print(e)
        return False

    return True


def quick_test_Zr(pot):
    pr = Project("/scratch/leimeroth/scratch/pyiron/projects/tmpZr")
    pr.remove_job("Interactive", _unprotect=True)
    try:
        lmp = pr.create.job.Lammps("Interactive")
        lmp.potential = pot
        lmp.structure = pr.create.structure.ase.bulk(
            "Zr", "fcc", a=4.53049372, cubic=True
        )
        lmp.server.run_mode.interactive = True
        lmp.interactive_enforce_structure_reset = True
        lmp.calc_minimize(pressure=np.zeros(3), max_iter=20)
        lmp.run()
        e_fcc = lmp.interactive_energy_pot_getter() / 4

        lmp.structure = pr.create.structure.ase.bulk(
            "Zr", "hcp", a=3.23480096, c=5.17229643
        )
        lmp.run()
        e_hcp = lmp.interactive_energy_pot_getter() / 2
        if e_fcc < e_hcp:
            return False

        lmp.structure = pr.create.structure.ase.bulk(
            "Zr", "sc", a=2.89618925, cubic=True
        )
        lmp.run()
        e_sc = lmp.interactive_energy_pot_getter()
        if e_sc < e_hcp:
            return False

        lmp.structure = pr.create.structure.ase.bulk(
            "Zr", "bcc", a=3.57436298, cubic=True
        )
        lmp.run()
        e_bcc = lmp.interactive_energy_pot_getter() / 2
        if e_bcc < e_hcp:
            return False

        lmp.structure = pr.create.structure.ase.bulk(
            "Zr", "diamond", a=6.11233701, cubic=True
        )
        lmp.run()
        e_dia = lmp.interactive_energy_pot_getter() / 8
        if e_dia < e_hcp:
            return False

    except Exception as e:
        print("Error in interactive job")
        print(e)
        return False

    return True


def quick_test_Li(pot):
    pr = Project("/scratch/leimeroth/scratch/pyiron/projects/tmpLi")
    pr.remove_job("Interactive", _unprotect=True)
    try:
        lmp = pr.create.job.Lammps("Interactive")
        lmp.potential = pot
        lmp.structure = pr.create.structure.ase.bulk("Li", a=3.426, cubic=True)
        lmp.server.run_mode.interactive = True
        lmp.interactive_enforce_structure_reset = True
        lmp.calc_minimize(pressure=np.zeros(3), max_iter=1000)
        lmp.run()
        e_bcc = lmp.interactive_energy_pot_getter() / 2

        lmp.structure = pr.create.structure.ase.bulk("Li", "hcp", a=3.13, c=5.02)
        lmp.run()
        e_hcp = lmp.interactive_energy_pot_getter() / 2
        if e_bcc - e_hcp > 0.003:
            print("hcp more stable")
            return False

        lmp.structure = pr.create.structure.ase.bulk("Li", "sc", a=2.75, cubic=True)
        lmp.run()
        e_sc = lmp.interactive_energy_pot_getter()
        if e_sc < e_bcc:
            print("sc more stable")
            return False

        lmp.structure = pr.create.structure.ase.bulk("Li", "fcc", a=4.4, cubic=True)
        lmp.run()
        e_fcc = lmp.interactive_energy_pot_getter() / 4
        if e_bcc - e_fcc > 0.003:
            print("fcc more stable")
            return False

        lmp.structure = pr.create.structure.ase.bulk(
            "Li", "diamond", a=5.92, cubic=True
        )
        lmp.run()
        e_dia = lmp.interactive_energy_pot_getter() / 8
        if e_dia < e_bcc:
            print("dia more stable")
            return False

    except Exception as e:
        print("Error in interactive job")
        print(e)
        return False

    return True

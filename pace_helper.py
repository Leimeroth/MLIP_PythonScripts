import os

import ase
import ase.data
import ase.db
import ase.io
import pandas as pd


import potential_testing as pot_test


def PACE_DF(
    path,
    pot_file_name="output_potential.yace",
    elements=["Cu"],
    algo="recursive",
    AL=False,
):
    elements_str = ""
    for e in elements:
        elements_str += f" {e}"

    if pot_file_name is None:
        fname = path
        pot_file_name = path.split("/")[-1]
    else:
        fname = f"{path}/{pot_file_name}"

    pot_file_name_ending = pot_file_name.split(".")[-1]

    if AL:
        if pot_file_name_ending != "yaml":
            raise ValueError("Use .yaml format for active learning")
        asi_file = pot_file_name.replace(".yaml", ".asi")
        config = [
            "pair_style pace/extrapolation\n",
            f"pair_coeff * * {pot_file_name} {asi_file} {elements_str}\n",
        ]
        files = [fname, f"{path}/{asi_file}"]

    else:
        if pot_file_name_ending == "yaml":
            raise ValueError("Convert yaml potential file to be compatible with lammps")
        config = [
            f"pair_style pace {algo}\n",
            f"pair_coeff * * {pot_file_name} {elements_str}\n",
        ]
        files = [fname]

    pot = pd.DataFrame(
        {
            "Name": "PACE",
            "Filename": [files],
            "Model": ["Custom"],
            "Species": [elements],
            "Config": [config],
        }
    )

    return pot


def collect_pace_al(f1, f2, specorder=["Si", "O", "C"], dbname=None):
    if dbname is None:
        dbname = "collected_AL.db"
    db = ase.db.connect(dbname)

    Z_of_type = {}
    for i in range(len(specorder)):
        Z_of_type[i + 1] = ase.data.atomic_numbers[specorder[i]]

    for i in range(f1, f2 + 1):
        base_atoms = ase.io.read(
            f"{i}/config.data",
            format="lammps-data",
            Z_of_type=Z_of_type,
            style="atomic",
        )
        base_numbers = base_atoms.numbers

        at_lst = ase.io.read(
            f"{i}/extrapolative_structures.dump",
            format="lammps-dump-text",
            index=":",
            order=True,
        )

        for at in at_lst:
            at.set_atomic_numbers(base_numbers)
            db.write(atoms=at)


class PacemakerData:
    def __init__(self, path):
        self.path = path
        self._read_train_metrics()
        self._read_test_metrics()

    @staticmethod
    def _read_metrics(metrics_file):
        return pd.read_csv(
            metrics_file,
            sep="\s+",
            header=0,
            index_col=False,
        )

    def _read_train_metrics(self):
        self.train_metrics = self._read_metrics(os.path.join(self.path, "metrics.txt"))

    def _read_test_metrics(self):
        self.test_metrics = self._read_metrics(
            os.path.join(self.path, "test_metrics.txt")
        )

    def plot_metrics(
        self,
        ycol="rmse_epa",
        xcol="iter_num",
        xlabel="Minimizer iteration",
        ylabel="RMSE [eV/atom]",
        fig_ax=None,
        every=1,
    ):
        fig, ax = pot_test._get_fig_ax(fig_ax)

        ax.plot(
            self.train_metrics[xcol][::every],
            self.train_metrics[ycol][::every],
            marker=None,
            color="black",
            label="train",
        )
        ax.plot(
            self.test_metrics[xcol][::every],
            self.test_metrics[ycol][::every],
            marker="o",
            color="red",
            label="test",
        )
        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        return fig, ax

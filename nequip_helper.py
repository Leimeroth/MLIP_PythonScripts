import pandas as pd


def NEQUIP_DF(path, pot_file_name=None, elements=["Al", "Cu", "Zr"], pot="nequip"):
    elements_str = ""
    for e in elements:
        elements_str += f" {e}"

    if pot_file_name is None:
        fname = path
        pot_file_name = path.split("/")[-1]
    else:
        fname = f"{path}/{pot_file_name}"

    ps_str = f"pair_style {pot}\n"

    config = [
        f"{ps_str}",
        f"pair_coeff * * {pot_file_name} {elements_str}\n",
    ]
    files = [fname]

    pot = pot = pd.DataFrame(
        {
            "Name": "NEQUIP",
            "Filename": [files],
            "Model": ["Custom"],
            "Species": [elements],
            "Config": [config],
        }
    )

    return pot

import os
import xml.etree.ElementTree as ET

import ase.data
import pandas as pd


def get_ini_str(potfile):
    tree = ET.parse(potfile)
    xml_element = tree.find('Potential')
    return xml_element.attrib['init_args']
    
def GAP_DF(path, pot_file, elements):
    elements_str = ""
    for e in elements:
        elements_str += f" {ase.data.atomic_numbers[e]}"

    files = [f for f in os.listdir(path) if f.startswith(pot_file)]
    files = [f'{path}/{f}' for f in files]

    ini_str = get_ini_str(os.path.join(path, pot_file))

    pot = pd.DataFrame(
        {
            "Name": "GAP",
            "Filename": [
                files
            ],
            "Model": ["Custom"],
            "Species": [elements],
            "Config": [
                [
                    'pair_style quip\n',
                    f'pair_coeff * * {pot_file} "{ini_str}" {elements_str}\n',
                ]
            ],
        }
    )
    return pot
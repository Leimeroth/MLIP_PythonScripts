#!/usr/bin/env python3 
import os
import pathlib

pot_paths = list(pathlib.Path(".").rglob("output_potential.yaml"))
paths = [path for path in pot_paths if not os.path.isfile(path.__str__().replace("yaml", "yace"))]

for path in paths:
    os.system(f"pace_yaml2yace {path}")

import pathlib
import sys

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets

data = datasets.get_dataset("villacampa_kidney_organoid", visualize=True)
breakpoint()
#DONE
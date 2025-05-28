import pathlib
import sys

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets

#data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
data = datasets.get_dataset("parigi_mouse_intestine", visualize=False)
breakpoint()
#DONE
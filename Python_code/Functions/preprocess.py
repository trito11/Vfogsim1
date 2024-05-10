import pandas as pd
import os
from pathlib import Path
import pandas as pd
link=Path(os.path.abspath(__file__))
link=link.parent.parent

def preprocess(file_addr, source):
    data_path = f'{link}/Data/{source}/{file_addr}'
    traffic = pd.read_csv(data_path).values
    return traffic
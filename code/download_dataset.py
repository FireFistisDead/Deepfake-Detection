"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
import zipfile

if __name__ == "__main__":
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
    cmd = f'curl -o LA.zip -# "{url}"'
    os.system(cmd)

    with zipfile.ZipFile("LA.zip", "r") as zip_ref:
        zip_ref.extractall(".")

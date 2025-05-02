from bs4 import BeautifulSoup
import base64
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from jupyter_client.kernelspec import get_kernel_spec
from jupyter_client.manager import AsyncKernelManager
from pydantic import Field
import os # Import os for path checking
import re
from typing import AsyncGenerator, Optional, List, Union # Ensure List and Union are imported


notebook_path = '../workspace/example.ipynb'
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)
print(nb)
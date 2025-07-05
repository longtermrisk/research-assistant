import sys
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

from utils import clean_ansi
from mcp.server.fastmcp import Image
from mcp.types import TextContent
from server import mcp



def replace_require_plotly(html_content):
    # Check if plotly require pattern exists in the HTML
    if 'require(["plotly"]' not in html_content:
        return html_content
        
    # Create BeautifulSoup object for HTML manipulation
    soup = BeautifulSoup(html_content, 'html.parser')
    # Get script tags
    script_tags = soup.find_all('script')
    script_content = script_tags[0].text
    modified_script_content = re.sub(r'require\(\["plotly"\], function\(Plotly\) {([\s\S]*)}\);', r'\1', script_content)
    # Replace the script content
    script_tags[0].string.replace_with(modified_script_content)

    # Add a script tag to load Plotly from CDN
    script_tag = soup.new_tag('script')
    script_tag['src'] = 'https://cdn.plot.ly/plotly-latest.min.js'
    script_tags[0].insert_before(script_tag)
    return str(soup)


def process_html(html_content):
    # Replace require(["plotly"], function(Plotly) {...}); with the Plotly script
    html_content = replace_require_plotly(html_content)
    # Add doctype, head, and body tags
    html_content = f"<!DOCTYPE html><html><head></head><body>{html_content}</body></html>"
    return html_content

def maybe_shorten(text):
    lines = text.split('\n')
    if len(lines) > 150:
        return "\n".join(lines[:50]) + f"\n... ({len(lines) - 100} lines skipped)" + "\n".join(lines[-50:])

# Helper function refactored from open_jupyter_notebook
def _format_notebook_to_blocks(nb: nbformat.NotebookNode, title: str) -> List[Union[str, Image]]:
    """Return an ordered list of `str` and `Image` blocks for the given notebook object."""
    blocks: List[Union[str, Image]] = []
    text_buf: List[str] = []

    def flush():
        if text_buf:
            blocks.append(maybe_shorten("".join(text_buf)))
            text_buf.clear()

    text_buf.append(f"{title}\n\n")

    for i, cell in enumerate(nb["cells"]):
        text_buf.append(f"## Cell {i+1}: {cell.cell_type}\n\n")

        if cell.cell_type == "code":
            text_buf.append("```python\n" + cell.source + "\n```\n")
            if cell.get("outputs"):
                text_buf.append("\n### Outputs:\n")
                for j, output in enumerate(cell.outputs):
                    text_buf.append(f"\n--- Output {j+1} ({output.output_type}) ---\n")
                    if output.output_type == "stream":
                        # Added strip() to avoid empty ```text blocks for just newlines
                        cleaned_text = output.get("text", "").strip()
                        if cleaned_text:
                            text_buf.append("```text\n" + cleaned_text + "\n```\n")
                        else:
                            text_buf.pop() # Remove the "--- Output..." line if stream is empty
                            text_buf.pop() # Remove the "### Outputs:" line if this was the only output
                    elif output.output_type == "error":
                        tb = "\n".join(output.traceback)
                        text_buf.append(f"**Error:** {output.ename}\n```traceback\n{clean_ansi(tb)}\n```\n") # Clean ANSI here too
                    elif output.output_type in {"display_data", "execute_result"}:
                        handled_img = False
                        # Check for PNG image data
                        if "image/png" in output.data:
                            flush()
                            img_bytes = base64.b64decode(output.data["image/png"])
                            blocks.append(Image(data=img_bytes, format="png"))
                            handled_img = True
                        # Check for plain text data, only if not handled as image
                        if "text/plain" in output.data and not handled_img:
                            # Added strip()
                            plain_text = output.data["text/plain"].strip()
                            if plain_text:
                                text_buf.append("```text\n" + plain_text + "```\n")
                        # Check for HTML data, only if not handled as image
                        if "text/html" in output.data and not handled_img:
                            flush()
                            html_content = output.data["text/html"]
                            blocks.append(TextContent(
                                type='text',
                                text="[HTML output not displayed]\n",
                                annotations={'display_html': process_html(html_content)},
                            ))

                    # Catch all for unhandled output types
                    else:
                         # Check if there's actually any content to display, otherwise skip
                        if output.get("data") or output.get("text"):
                            text_buf.append(f"[Unhandled output type: {output.output_type}]\n")
                        else:
                            # If no data/text, remove the "--- Output..." line
                            text_buf.pop()
                            # If this was the only output, remove "### Outputs:" too
                            if not cell.outputs or j == 0:
                                text_buf.pop()


            # If after processing all outputs for a code cell, there were none, remove the "### Outputs:" header
            if text_buf and text_buf[-1] == "\n### Outputs:\n":
                text_buf.pop()

        elif cell.cell_type == "markdown":
            text_buf.append(cell.source + "\n")
        elif cell.cell_type == "raw":
            text_buf.append("```raw\n" + cell.source + "```\n")
        else:
            text_buf.append(f"[Unsupported cell type: {cell.cell_type}]\n")
        text_buf.append("\n") # Add space between cells

    flush()
    return blocks


# Keep the original function signature for compatibility
def open_jupyter_notebook(notebook_path: str) -> List[Union[str, Image]]:
    """Return an ordered list of `str` and `Image` blocks for the notebook by reading from path."""
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as exc:
        return [f"Error reading notebook {os.path.basename(notebook_path)}: {exc}"]
    # Use relative path for title if possible, otherwise just basename
    try:
        rel_path = os.path.relpath(notebook_path, os.getcwd())
    except ValueError: # Handles cases like different drives on Windows
        rel_path = os.path.basename(notebook_path)

    title = f"# Jupyter Notebook: {rel_path}"
    return _format_notebook_to_blocks(nb, title)


class JupyterNotebook:
    """A wrapper around a Jupyter notebook that can execute code and return results."""
    def __init__(self, path: Optional[str] = None):
        """Initialize the Jupyter notebook."""
        self.path = path # Store path for reference
        self.notebook = None
        
        # Get the venv Python executable from the current working directory
        venv_python = os.path.abspath(os.path.join(os.getcwd(), ".venv/bin/python"))
        if not os.path.exists(venv_python):
            raise ValueError(f"No venv Python found at {venv_python}. Please ensure a virtual environment exists in the current working directory.")
            
        # Create a unique kernel name based on the cwd to avoid conflicts
        cwd_str = os.getcwd().replace('/', '-').replace('.', '')
        cwd_hash = sum(ord(c) * (i + 1) for i, c in enumerate(cwd_str)) % 10000
        kernel_name = f"venv_kernel_{cwd_hash}"
        
        try:
            # Check if kernel already exists
            from jupyter_client import kernelspec
            kernelspec.get_kernel_spec(kernel_name)
        except:
            # Register new kernel using the venv Python
            import subprocess
            import json
            import tempfile
            import shutil
            
            # Create a temporary directory for the kernel spec
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create kernel.json
                kernel_json = {
                    "argv": [venv_python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": f"Venv Kernel ({cwd_hash})",
                    "language": "python",
                    "metadata": {"debugger": True}
                }
                
                # Write kernel.json to the temp directory
                kernel_json_path = os.path.join(temp_dir, "kernel.json")
                with open(kernel_json_path, "w") as f:
                    json.dump(kernel_json, f)
                
                # Install the kernel spec
                try:
                    subprocess.run([
                        venv_python, "-m", "jupyter", "kernelspec", "install",
                        "--user", "--name", kernel_name,
                        "--replace",  # Replace if exists
                        temp_dir
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install kernel spec: {e}")
                    raise
        
        # Load or create notebook (rest of the initialization)
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.notebook = nbformat.read(f, as_version=4)
            print(f"Loaded notebook from {path}")
        else:
            self.notebook = nbformat.v4.new_notebook()
            if path:
                print(f"Path {path} not found. Created a new notebook.")
            else:
                print("Created a new temporary notebook.")

        self.km = None
        self.kc = None
        self.client = NotebookClient(
            self.notebook,
            timeout=600,
            allow_errors=True,
            kernel_name=kernel_name,
        )
        
        # Set working directory for the kernel
        if path:
            self.client.resources = {'metadata': {'path': os.path.dirname(os.path.abspath(path))}}
        else:
            self.client.resources = {'metadata': {'path': os.getcwd()}}

    async def setup(self):
        """Initialize the kernel if not already initialized."""
        if self.client.kc is None:
            # Create and start kernel manager if not exists
            if self.km is None:
                self.km = AsyncKernelManager(kernel_name=self.client.kernel_name)  # Use same kernel name as client
                await self.km.start_kernel(cwd=self.client.resources['metadata']['path'])
         
            # Assign the kernel manager to the client
            # Note: nbclient >= 0.8.0 prefers setting km directly
            # For older versions you might need self.client.km = self.km
            # Let's try setting kc directly from km which is more robust
            self.kc = self.km.client()
            self.client.kc = self.kc

            # Ensure the kernel client is started and ready
            # No explicit start_new_kernel_client needed if kc is assigned
            # await self.client.async_start_new_kernel_client() # Might not be needed
            while not await self.kc.is_alive():
                 await asyncio.sleep(0.1)
            print("Kernel is ready.")


    async def execute_cell(self, cell: nbformat.NotebookNode) -> None:
        """Executes a single cell within the notebook."""
        # Find the index dynamically - crucial if cells were added/removed
        try:
            cell_index = self.notebook.cells.index(cell)
            print(f"Executing cell {cell_index + 1}/{len(self.notebook.cells)}")
            await self.client.async_execute_cell(cell, cell_index)
        except ValueError:
            print("Error: Cell not found in the notebook's cell list.")
        except CellExecutionError as e:
            # Error is already stored in cell output by allow_errors=True
            print(f"Error executing cell: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during cell execution: {e}")
            # Potentially add error info to the cell manually if needed
            error_output = nbformat.v4.new_output(
                output_type='error',
                ename=type(e).__name__,
                evalue=str(e),
                traceback=[str(e)]
            )
            if not hasattr(cell, 'outputs'):
                cell.outputs = []
            cell.outputs.append(error_output)

    async def execute_all_cells(self) -> None:
        """Execute all existing cells in the notebook sequentially."""
        if not self.notebook.cells:
            print("Notebook has no cells to execute.")
            return

        print(f"Executing all {len(self.notebook.cells)} existing cells...")
        # Ensure kernel is initialized before executing multiple cells
        await self.setup()

        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'code':
                await self.execute_cell(cell)
            # else: # Optionally print info about non-code cells
            #     print(f"Skipping non-code cell {i+1} ({cell.cell_type})")
        print("Finished executing all existing cells.")


    async def execute_new_code(self, code: str) -> List[Union[TextContent, Image]]:
        """Append a new code cell, execute it, and return its outputs."""
        # Ensure kernel is initialized
        await self.setup()

        # Create and append the new code cell
        cell = nbformat.v4.new_code_cell(code)
        self.notebook.cells.append(cell)
        print(f"Appended new cell {len(self.notebook.cells)}")

        # Execute the newly added cell
        await self.execute_cell(cell)

        # Process outputs from the *newly executed cell*
        results: List[Union[TextContent, Image]] = []
        
        if not hasattr(cell, 'outputs'):
            print("Cell finished execution, no outputs generated.")
            return results

        print(f"Processing {len(cell.outputs)} outputs for the new cell...")
        for output in cell.outputs:
            output_type = output.get('output_type', '')

            if output_type == 'error':
                # Handle error output
                error_msg = '\n'.join(output.get('traceback', []))
                results.append(TextContent(
                    type='text',
                    text=clean_ansi(f"Error: {output.ename}\n{error_msg}")
                ))

            elif output_type == 'stream':
                # Handle stdout/stderr, clean ANSI codes
                stream_text = output.get('text', '')
                cleaned_text = maybe_shorten(clean_ansi(stream_text).strip())
                if cleaned_text: # Add only if there's non-empty text
                    results.append(TextContent(type='text', text=cleaned_text))

            elif output_type == 'display_data' or output_type == 'execute_result':
                data = output.get('data', {})

                # Handle images (PNG)
                if 'image/png' in data:
                    img_data = data['image/png']
                    # Decode if it's base64 encoded string
                    if isinstance(img_data, str):
                        try:
                            img_data = base64.b64decode(img_data)
                            results.append(Image(data=img_data, format='png'))
                        except Exception as e:
                            results.append(TextContent(type='text', text=f"[Error decoding base64 PNG: {e}]"))
                    elif isinstance(img_data, bytes):
                         results.append(Image(data=img_data, format='png'))
                    continue # Don't process other formats if image was shown

                # Handle HTML
                elif 'text/html' in data:
                    html_content = data['text/html']
                    results.append(TextContent(
                        type='text',
                        text="[HTML output not displayed]\n",
                        annotations={'display_html': process_html(html_content)},
                    ))
                    continue # Don't process plain text if HTML was primary

                # Handle plain text as fallback
                elif 'text/plain' in data:
                    text_content = data['text/plain']
                    if isinstance(text_content, list): # Sometimes it's a list of strings
                        text_content = '\n'.join(text_content)
                    cleaned_text = maybe_shorten(clean_ansi(text_content).strip())
                    if cleaned_text: # Add only if there's non-empty text
                         results.append(TextContent(type='text', text=cleaned_text))
            else: # Optionally log unhandled types
                 results.append(TextContent(type='text', text=f"Note: Unhandled output type '{output_type}' encountered."))
        
        return results

    def save(self):
        """Save the notebook to the specified path."""
        if self.path:
            try:
                with open(self.path, 'w', encoding='utf-8') as f:
                    nbformat.write(self.notebook, f)
                print(f"Notebook saved to {self.path}")
            except Exception as e:
                print(f"Error saving notebook to {self.path}: {e}")
        else:
            print("No path specified for saving the notebook.")

    async def cleanup(self):
        """Clean up resources when done."""
        if self.client and self.client.kc:
            print("Shutting down kernel client...")
            try:
                # Use wait=True for cleaner shutdown
                await self.client.kc.stop_channels()
                # Shutdown the kernel managed by the kernel manager
                if self.km and self.km.has_kernel:
                     print("Shutting down kernel...")
                     await self.km.shutdown_kernel(now=True) # 'now=True' forces immediate shutdown
                     self.km = None # Clear reference
                self.kc = None # Clear reference
                self.client.kc = None # Ensure client knows kernel is gone
            except Exception as e:
                print(f"Error during kernel cleanup: {e}")
        elif self.km and self.km.has_kernel: # Fallback if client wasn't fully setup
            print("Shutting down kernel manager directly...")
            await self.km.shutdown_kernel(now=True)
            self.km = None


# In-memory store for active notebooks, using None key for the temporary one
# Path (if provided) acts as the key otherwise
_OPEN_NOTEBOOKS: dict[Optional[str], JupyterNotebook] = {}

# Import asyncio for setup check
import asyncio

@mcp.tool()
async def jupyter(
    code: str = Field(..., description="The Python code to execute in the notebook."),
    path: Optional[str] = Field(None, description="Path to a .ipynb file. If provided and exists, existing cells are run on first call. If provided but doesn't exist, a new notebook is created at this key. If omitted, a temporary stateful notebook is used.")
) -> List[Union[str, Image]]:
    """
    Run Python code in a Jupyter notebook environment. Maintains state per notebook path.

    - If `path` is omitted, code runs in a persistent temporary notebook session.
    - If `path` is provided for the first time:
        - If the file exists, it's loaded, and *all* its cells are executed sequentially.
        - If the file doesn't exist, a new empty notebook is created (associated with this path).
        - Then, the provided `code` is added as a new cell and executed.
        - The output returned is the formatted content of the *entire* notebook after execution.
    - If `path` is provided again (for a path already opened in this session):
        - The provided `code` is added as a new cell to the existing notebook state and executed.
        - The output returned contains *only* the results (text, images) from executing this new cell.

    Args:
        code: The Python code to execute.
        path: The path to the notebook file (.ipynb).

    Returns:
        A list containing strings (formatted text, code, errors) and Image objects representing the output.
        The content depends on whether it's the first call with a specific path.
    """
    # Use None as the key for the temporary notebook if path is not provided
    notebook_key = path

    outputs: List[Union[str, Image]] = []
    notebook: Optional[JupyterNotebook] = None
    is_first_run_with_path = False

    if notebook_key in _OPEN_NOTEBOOKS:
        print(f"Using existing notebook session for key: {notebook_key}")
        notebook = _OPEN_NOTEBOOKS[notebook_key]
        # Ensure kernel is still alive, setup if necessary (e.g., after inactivity timeout)
        if not notebook.km or not await notebook.km.is_alive():
            print("Kernel needs setup/restart.")
            await notebook.setup()
    else:
        print(f"Creating new notebook session for key: {notebook_key}")
        # Check if the path exists *before* creating the JupyterNotebook instance
        # This determines if we need to run existing cells
        path_existed = path is not None and os.path.exists(path)

        try:
            notebook = JupyterNotebook(path=path)
            _OPEN_NOTEBOOKS[notebook_key] = notebook
            # Perform initial kernel setup
            await notebook.setup()

            # If the path existed and we successfully loaded cells, run them all
            if path_existed and notebook.notebook.cells:
                is_first_run_with_path = True # Mark this for later output formatting
                print(f"Path '{path}' exists, executing existing cells...")
                await notebook.execute_all_cells()
            elif path: # Path provided but didn't exist or failed load (empty notebook created)
                is_first_run_with_path = True # Treat as first run for this path key
                print(f"New notebook created for path '{path}'.")
            # No need to run existing cells if temporary or new file path
        except ValueError as e:
            return [f"Error initializing notebook: {e}"]
        except Exception as e:
            # Catch unexpected errors during setup or initial run
            import traceback
            tb_str = traceback.format_exc()
            # Clean up kernel if partially started
            if notebook: await notebook.cleanup()
            if notebook_key in _OPEN_NOTEBOOKS: del _OPEN_NOTEBOOKS[notebook_key]
            return [f"Unexpected error setting up notebook for key '{notebook_key}': {e}\n{tb_str}"]


    # Now, execute the *new* code provided in the call
    print(f"Executing provided code in notebook for key: {notebook_key}...")
    try:
        new_cell_outputs = await notebook.execute_new_code(code)
    except Exception as e:
        # Catch unexpected errors during the execution of the new code
        import traceback
        tb_str = traceback.format_exc()
        # Append error to outputs
        new_cell_outputs.append(f"Unexpected error executing new code: {e}\n{tb_str}")
    
    # If a path was provided, we need to save the notebook
    if path:
        try:
            notebook.save()
            print(f"Notebook saved to {path}")
        except Exception as e:
            print(f"Failed to save notebook to {path}: {e}")

    # Determine what to return
    if is_first_run_with_path:
        # Return the formatted content of the entire notebook
        print("First run with path, returning entire notebook content.")
        title = f"# Jupyter Notebook: {path}" # Use the provided path in the title
        # Ensure the notebook object passed to formatting is the updated one
        return _format_notebook_to_blocks(notebook.notebook, title)
    else:
        # Return only the outputs of the newly executed cell
        print("Returning output from the latest executed cell.")
        return new_cell_outputs


# Example of how to potentially clean up all notebooks on server shutdown
# This would need integration into your MCP server's lifecycle management
async def cleanup_all_notebooks():
     print("Cleaning up all active Jupyter notebooks...")
     for key, notebook in list(_OPEN_NOTEBOOKS.items()): # Iterate over copy of items
        print(f"Cleaning up notebook for key: {key}")
        await notebook.cleanup()
        del _OPEN_NOTEBOOKS[key]
     print("Finished cleaning up notebooks.")

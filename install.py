import subprocess
import os
import json
import shutil


def ensure_uv():
    """
    Ensure that the 'uv' module is installed. If not, install it.
    """
    # Run which uv
    try:
        subprocess.run(['which', 'uv'], check=True)
        print("uv is already installed.")
        return
    except subprocess.CalledProcessError:
        print("uv is not installed. Installing...")
        # Install uv:
    # - macos/linus: curl -LsSf https://astral.sh/uv/install.sh | sh
    try:
        subprocess.run(['curl', '-LsSf', 'https://astral.sh/uv/install.sh', '|', 'sh'], check=True)
        print("uv installed successfully.")
    except subprocess.CalledProcessError:
        input("Failed to install uv. Please install it manually and click Enter to continue.")


def setup_mcp_dot_json():
    """Copies mcp.json from the __file__ directory to the ~/mcp.json
    
    Replaces:
    - '/path/to/uv' with the actual path to uv
    - '/path/to/repo/' with the actual path to the repository
    - all env's with the actual envs or override by input
    """
    if os.path.exists(os.path.expanduser('~/mcp.json')):
        with open(os.path.expanduser('~/mcp.json'), 'r') as f:
            existing_mcp_json = json.load(f)
    else:
        existing_mcp_json = {}

    with open(os.path.join(os.path.dirname(__file__), 'mcp.json'), 'r') as f:
        mcp_json = f.read()
    
    # Replace '/path/to/uv' with the actual path to uv
    uv_path = subprocess.run(['which', 'uv'], capture_output=True, text=True).stdout.strip()
    mcp_json = mcp_json.replace('/path/to/uv', uv_path)
    
    # Replace '/path/to/repo/' with the actual path to the repository
    repo_path = os.path.dirname(os.path.abspath(__file__))
    mcp_json = mcp_json.replace('/path/to/repo', repo_path)

    mcp_data = json.loads(mcp_json)
    confirmed_env = {}
    for name, config in mcp_data['mcpServers'].items():
        print(f"Setting up {name}...")
        if 'env' in config:
            env = config['env']
            existing_env = {}
            if name in existing_mcp_json['mcpServers']:
                existing_env = existing_mcp_json['mcpServers'][name].get('env', {})
                for key, value in existing_env.items():
                    if key in env:
                        env[key] = value
            for key, value in env.items():
                if key in existing_env:
                    value = existing_env[key]
                    continue
                if key in confirmed_env:
                    value = confirmed_env[key]
                else:
                    default = os.environ.get(key)
                    value = input(f"Please enter the value for {key} (default: {default}): ")
                    if not value:
                        value = default
                    confirmed_env[key] = value
                env[key] = value
    
    for name, config in existing_mcp_json['mcpServers'].items():
        if name not in mcp_data['mcpServers']:
            mcp_data['mcpServers'][name] = config
    
    # Write the modified mcp.json to ~/mcp.json
    with open(os.path.expanduser('~/mcp.json'), 'w') as f:
        json.dump(mcp_data, f, indent=4)
    print("mcp.json has been set up in ~/mcp.json.")


def setup_component(relpath):
    """
    Setup mcp server:
        cd relpath
        uv venv
        source .venv/bin/activate
        uv sync
    """
    path = os.path.join(os.path.dirname(__file__), relpath)
    os.chdir(path)
    if not os.path.exists('.venv'):
        subprocess.run(['uv', 'venv'], check=True)
    subprocess.run(['uv', 'sync'], check=True)

def setup_frontend():
    """
    Setup frontend:
        cd frontend
        npm install
    """
    path = os.path.join(os.path.dirname(__file__), 'automator/frontend')
    os.chdir(path)
    subprocess.run(['npm', 'install'], check=True)


def get_env_from_mcp_json():
    """
    Get the env from mcp.json
    """
    with open(os.path.expanduser('~/mcp.json'), 'r') as f:
        mcp_json = json.load(f)
    
    env = {}
    for name, config in mcp_json['mcpServers'].items():
        if 'env' in config:
            env.update(config['env'])

    # Now write to .env
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    with open(env_path, 'w') as f:
        for key, value in env.items():
            f.write(f"{key}={value}\n")
    print(f"Environment variables have been written to {env_path}.")


def start_backend():
    """
    Start backend:
        cd backend
        .venv/bin/python -m uvicorn automator.api.main:app --port 8000 &> ../backend.logs (in the background)
    """
    path = os.path.join(os.path.dirname(__file__), 'automator')
    os.chdir(path)
    python_path = os.path.join(path, '.venv', 'bin', 'python')
    subprocess.Popen(
        [python_path, '-m', 'uvicorn', 'automator.api.main:app', '--port', '8000'],
        stdout=open('../backend.logs', 'a'),
        stderr=subprocess.STDOUT
    )
    print("Backend started. Logs are being written to backend.logs.")

def start_frontend():
    """
    Start frontend:
        cd frontend
        npm start &> ../frontend.logs(in the background)
    """
    path = os.path.join(os.path.dirname(__file__), 'automator/frontend')
    os.chdir(path)
    subprocess.Popen(['npm', 'run', 'dev'], stdout=open('../../frontend.logs', 'a'), stderr=subprocess.STDOUT)
    print("Frontend started. Logs are being written to frontend.logs.")
    print("You can now access the frontend at http://localhost:5173.")

def start_all():
    """
    Start all components:
        start_backend
        start_frontend
    """
    start_backend()
    start_frontend()


def install_repo():
    """
    Installs all components of the repo
    """
    ensure_uv()
    setup_mcp_dot_json()
    os.makedirs(os.path.expanduser('~/.automator/workspaces'), exist_ok=True)
    # cp prompts ~/.automator/prompts
    shutil.copytree(os.path.join(os.path.dirname(__file__), 'automator/prompts'), os.path.expanduser('~/.automator/prompts'), dirs_exist_ok=True)
    setup_component('terminal-mcp')
    setup_component('web-mcp')
    setup_component('talk-to-model')
    setup_component('automator')
    setup_frontend()
    if input("Do you want to start all components? (y/n): ").lower() == 'y':
        start_all()
    print("For more examples, checkout examples/ and more_examples/")


if __name__ == "__main__":
    install_repo()

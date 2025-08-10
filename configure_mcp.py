import json
import os
import sys
from pathlib import Path

def configure_mcp():
    """
    Generates /root/mcp.json from a template and environment variables.
    Also creates an /app/.env file for compatibility.
    """
    mcp_template_path = Path('/app/mcp.json')
    mcp_config_path = Path.home() / 'mcp.json'
    repo_path = '/app'
    
    try:
        uv_path = os.popen('which uv').read().strip()
        if not uv_path:
            raise FileNotFoundError("'uv' command not found.")
    except Exception as e:
        print(f"Error finding uv path: {e}", file=sys.stderr)
        sys.exit(1)

    # Define required API keys
    required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'SERP_API_KEY', 'HF_TOKEN']
    
    # Check if all required environment variables are set
    missing_keys = [key for key in required_keys if key not in os.environ]
    if missing_keys:
        print(f"Error: Missing required environment variables: {', '.join(missing_keys)}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load the template mcp.json
        with open(mcp_template_path, 'r') as f:
            mcp_data = json.load(f)

        # Use string replacement for paths for simplicity
        mcp_data_str = json.dumps(mcp_data)
        mcp_data_str = mcp_data_str.replace('/path/to/repo', repo_path)
        mcp_data_str = mcp_data_str.replace('/path/to/uv', uv_path)
        mcp_data = json.loads(mcp_data_str)

        # Populate the 'env' section for each server with environment variables
        for server_config in mcp_data.get('mcpServers', {}).values():
            if 'env' in server_config:
                for key in server_config['env']:
                    if key in os.environ:
                        server_config['env'][key] = os.environ[key]

        # Write the final configuration file
        with open(mcp_config_path, 'w') as f:
            json.dump(mcp_data, f, indent=4)
        print(f"Successfully configured {mcp_config_path}")

        # Create a .env file in the app directory for other potential scripts
        with open(Path(repo_path) / '.env', 'w') as f:
            for key in required_keys:
                if key in os.environ:
                    f.write(f'{key}={os.environ[key]}\n')
        print(f"Successfully created {Path(repo_path) / '.env'}")

    except FileNotFoundError:
        print(f"Error: MCP template not found at {mcp_template_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {mcp_template_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    configure_mcp()
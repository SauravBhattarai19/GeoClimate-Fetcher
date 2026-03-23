"""
Helper script to set up local service account authentication.

Usage:
    python setup_service_account.py path/to/your-service-account.json

This creates .streamlit/secrets.toml so the Quick Access login works locally.
"""

import json
import sys
import os


def setup(json_path):
    # Validate the file exists
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Load and validate the JSON
    try:
        with open(json_path, 'r') as f:
            sa_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)

    # Check required fields
    required = ['type', 'project_id', 'client_email', 'private_key']
    missing = [k for k in required if k not in sa_data]
    if missing:
        print(f"Error: Missing required fields: {', '.join(missing)}")
        sys.exit(1)

    if sa_data['type'] != 'service_account':
        print(f"Error: Expected type 'service_account', got '{sa_data['type']}'")
        sys.exit(1)

    # Create .streamlit directory
    os.makedirs('.streamlit', exist_ok=True)

    # Build secrets.toml
    # key_json must be the entire JSON as a single escaped string
    key_json_str = json.dumps(sa_data)

    secrets_content = (
        "[gee]\n"
        f'client_email = "{sa_data["client_email"]}"\n'
        f'project_id = "{sa_data["project_id"]}"\n'
        f"key_json = '{key_json_str}'\n"
    )

    secrets_path = os.path.join('.streamlit', 'secrets.toml')

    # Warn if file already exists
    if os.path.exists(secrets_path):
        resp = input(f"{secrets_path} already exists. Overwrite? [y/N]: ").strip().lower()
        if resp != 'y':
            print("Aborted.")
            sys.exit(0)

    with open(secrets_path, 'w') as f:
        f.write(secrets_content)

    print(f"Done! Created {secrets_path}")
    print(f"Service account: {sa_data['client_email']}")
    print(f"Project: {sa_data['project_id']}")
    print()
    print("You can now use 'Quick Access' when running the app locally.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python setup_service_account.py <path-to-service-account.json>")
        sys.exit(1)

    setup(sys.argv[1])

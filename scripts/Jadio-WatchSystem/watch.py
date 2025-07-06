import os
import sys
import subprocess
import yaml
from pathlib import Path

def get_bootstrap_map():
   folder = os.path.dirname(os.path.abspath(__file__))
   bt_path = os.path.normpath(os.path.join(folder, 'watch_system', 'registry', 'bootstrap.txt'))
   def parse_bootstrap(path):
       mapping = {}
       with open(path, 'r', encoding='utf-8') as f:
           for line in f:
               line = line.strip()
               if not line or ':' not in line:
                   continue
               key, val = line.split(':', 1)
               mapping[key.strip()] = val.strip()
       return mapping
   return parse_bootstrap(bt_path)

def load_port_registry():
   bootstrap_map = get_bootstrap_map()
   port_registry_path = bootstrap_map['PORT_REGISTRY_PATH']
   
   with open(port_registry_path, 'r', encoding='utf-8') as f:
       return yaml.safe_load(f) or {}

def save_port_registry(registry):
   bootstrap_map = get_bootstrap_map()
   port_registry_path = bootstrap_map['PORT_REGISTRY_PATH']
   
   with open(port_registry_path, 'w', encoding='utf-8') as f:
       yaml.dump(registry, f, sort_keys=True, allow_unicode=True)

def get_next_available_port():
   registry = load_port_registry()
   used_ports = set(registry.values())
   port = 53010
   while port in used_ports:
       port += 1
   return port

def cli():
   while True:
       print("\n--- DataLabs Watch System ---")
       print("1. Set cleaner max lines")
       print("2. Set pinger interval")
       print("3. Run server")
       print("4. Stop server")
       print("5. Server status")
       print("6. Add script to port registry")
       print("7. Remove script from port registry")
       print("8. List all port assignments")
       print("9. Generate AI reference")
       print("10. Exit")
       choice = input("Select an option: ").strip()
       
       if choice == '1':
           lines = input("Enter max lines for cleaner: ").strip()
           subprocess.run([sys.executable, get_registry_script('CONFIG_PY'), 'cleaner', lines])
       elif choice == '2':
           interval = input("Enter pinger interval (seconds): ").strip()
           subprocess.run([sys.executable, get_registry_script('CONFIG_PY'), 'pinger', interval])
       elif choice == '3':
           subprocess.run([sys.executable, get_registry_script('START')])
       elif choice == '4':
           subprocess.run([sys.executable, get_registry_script('STOP')])
       elif choice == '5':
           show_status_log()
       elif choice == '6':
           add_script_to_registry()
       elif choice == '7':
           remove_script_from_registry()
       elif choice == '8':
           list_port_assignments()
       elif choice == '9':
           subprocess.run([sys.executable, get_registry_script('REFERENCE_DUMPER')])
       elif choice == '10':
           print("Exiting.")
           break
       else:
           print("Invalid option.")

def add_script_to_registry():
   script_path = input("Enter path to Python script: ").strip()
   registry = load_port_registry()
   script_name = Path(script_path).stem.upper()
   port_key = f"{script_name}_PORT"
   
   if port_key in registry:
       print(f"Script already registered: {port_key} -> Port {registry[port_key]}")
       return
   
   new_port = get_next_available_port()
   registry[port_key] = new_port
   save_port_registry(registry)
   print(f"Added {script_name} on port {new_port}")

def remove_script_from_registry():
   script_name = input("Enter script name to remove: ").strip()
   registry = load_port_registry()
   port_key = f"{script_name.upper()}_PORT"
   
   if port_key not in registry:
       print(f"Script not found: {script_name}")
       return
   
   removed_port = registry[port_key]
   del registry[port_key]
   save_port_registry(registry)
   print(f"Removed {script_name} (was on port {removed_port})")

def list_port_assignments():
   registry = load_port_registry()
   
   print("\nCurrent Port Assignments:")
   system_ports = {k: v for k, v in registry.items() if v < 53010}
   user_ports = {k: v for k, v in registry.items() if v >= 53010}
   
   if system_ports:
       print("\nSystem Ports:")
       for key, port in sorted(system_ports.items(), key=lambda x: x[1]):
           print(f"   {key}: {port}")
   
   if user_ports:
       print("\nUser Ports:")
       for key, port in sorted(user_ports.items(), key=lambda x: x[1]):
           print(f"   {key}: {port}")
   else:
       print("\nUser Ports: None assigned")
   
   print(f"\nNext available port: {get_next_available_port()}")

def get_registry_script(stub):
   bootstrap_map = get_bootstrap_map()
   script_path = bootstrap_map[stub]
   return script_path

def show_status_log():
   bootstrap_map = get_bootstrap_map()
   STATUS_LOG_PATH = bootstrap_map['STATUS_LOG_PATH']
   
   with open(STATUS_LOG_PATH, 'r', encoding='utf-8') as f:
       lines = f.readlines()
       print('--- Last 5 lines of status log ---')
       for line in lines[-5:]:
           print(line.rstrip())

if __name__ == '__main__':
   cli()
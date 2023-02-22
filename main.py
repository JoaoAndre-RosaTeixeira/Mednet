
import subprocess
import auto_pip_install


auto_pip_install.run_pip_install()
subprocess.run(["streamlit", "run", "home.py"])
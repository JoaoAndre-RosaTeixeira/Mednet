import subprocess
import sys


def run_pip_install():
    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


    pips_install = ['streamlit', 'pillow']

    for pip in pips_install:
        try:
            install(pip)
        except:
            print(f"impossible d'installer {pip}")





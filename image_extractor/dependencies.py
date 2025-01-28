import subprocess
import sys

# Lista de dependências a serem instaladas
dependencies = [
    "python-dotenv",
    "pandas",
    "python-Levenshtein",
    "langchain-google-vertexai==2.0.11",
    "torcheval"
]

def install_packages():
    for package in dependencies:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Error: {e}")

if __name__ == "__main__":
    install_packages()
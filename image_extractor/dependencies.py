import subprocess
import sys

# dependencies to install
dependencies = [
    "python-dotenv",
    "pandas",
    "python-Levenshtein",
    "langchain-google-vertexai==2.0.11",
    "torcheval",
    "google-cloud-vision",
    "langchain-anthropic",
    "langchain-ollama",
    "scikit-learn"
]


def install_packages():
    for package in dependencies:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Error: {e}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--pre",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/nightly/cpu",
        ]
    )


if __name__ == "__main__":
    install_packages()

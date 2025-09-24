import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        'flask',
        'flask-cors',
        'pandas',
        'numpy',
        'scikit-learn',
        'nltk',
        'requests'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def run_server():
    """Run the Flask server"""
    print("Installing required packages...")
    install_requirements()
    
    print("\nStarting Flask server...")
    print("Server will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("Prediction endpoint: http://localhost:5000/predict")
    print("\nPress Ctrl+C to stop the server")
    
    # Run the prediction server
    os.system("python scripts/predict.py")

if __name__ == "__main__":
    run_server()

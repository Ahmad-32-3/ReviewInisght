# Setup script to register the virtual environment as a Jupyter kernel
# Run this once after creating your virtual environment

Write-Host "Setting up Jupyter kernel for ReviewInsight..." -ForegroundColor Green

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Install ipykernel if not already installed
Write-Host "Installing ipykernel..." -ForegroundColor Yellow
python -m pip install ipykernel

# Register the kernel
Write-Host "Registering kernel..." -ForegroundColor Yellow
python -m ipykernel install --user --name=reviewinsight --display-name="Python (ReviewInsight)"

Write-Host "`nSetup complete! You can now select 'Python (ReviewInsight)' as your kernel in Jupyter notebooks." -ForegroundColor Green
Write-Host "In Cursor: Click 'Select Kernel' in the top right of your notebook and choose 'Python (ReviewInsight)'" -ForegroundColor Cyan


# PowerShell install helper (Windows)
Write-Host "This script creates a venv and installs requirements (user must install PyTorch manually if needed)."
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Note: Install PyTorch separately per https://pytorch.org to select CPU/GPU builds suitable for your machine." 

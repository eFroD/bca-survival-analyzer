# PowerShell script to set up pre-commit hooks

# Check if pre-commit is installed
$precommitInstalled = $null
try {
    $precommitInstalled = Get-Command pre-commit -ErrorAction SilentlyContinue
} catch {
    # Command not found
}

if ($null -eq $precommitInstalled) {
    Write-Host "Installing pre-commit..." -ForegroundColor Yellow
    pip install pre-commit
} else {
    Write-Host "pre-commit already installed." -ForegroundColor Green
}

# Check if package is installed in dev mode
$packageInstalled = $null
try {
    $packageInstalled = pip show bca-survival -ErrorAction SilentlyContinue
} catch {
    # Package not found
}

if ($null -eq $packageInstalled) {
    Write-Host "Installing package in development mode..." -ForegroundColor Yellow
    pip install -e ".[dev]"
} else {
    Write-Host "Package already installed in development mode." -ForegroundColor Green
}

# Install pre-commit hooks
Write-Host "Installing pre-commit hooks..." -ForegroundColor Yellow
pre-commit install

Write-Host "`nPre-commit hooks set up successfully!" -ForegroundColor Green
Write-Host "Hooks will run automatically on git commit."
Write-Host "To run hooks manually on all files: pre-commit run --all-files"
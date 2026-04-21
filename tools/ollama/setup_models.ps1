# Setup script for custom Ollama models
Write-Host "Creating custom Ollama models..." -ForegroundColor Cyan

# Change directory to the location of Modelfiles
$dir = Split-Path $MyInvocation.MyCommand.Path
Push-Location $dir

try {
    Write-Host "Creating claude-code-7b (Optimized for 8-Core CPU)..." -ForegroundColor Yellow
    ollama create claude-code-7b -f claude-code-7b.modelfile

    Write-Host "Creating claude-code-14b..." -ForegroundColor Yellow
    ollama create claude-code-14b -f claude-code-14b.modelfile

    Write-Host "Creating claude-code-32b..." -ForegroundColor Yellow
    ollama create claude-code-32b -f claude-code-32b.modelfile

    Write-Host "Models created successfully!" -ForegroundColor Green
}
finally {
    Pop-Location
}

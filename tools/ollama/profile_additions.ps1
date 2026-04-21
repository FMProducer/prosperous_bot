# Copy and paste these functions into your PowerShell profile ($PROFILE)
# You can open it with: notepad $PROFILE

function claude {
    ollama run claude-code-7b
}

# Alias for backward compatibility or specific versioning
function claude7 {
    ollama run claude-code-7b
}

function claude14 {
    ollama run claude-code-14b
}

Write-Host "Aliases 'claude' and 'claude7' (7B Optimized for 8-Core) are ready for use." -ForegroundColor Green

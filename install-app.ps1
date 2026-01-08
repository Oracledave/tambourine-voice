<#
PowerShell helper: Install dependencies (optional) and build the app from a repository.

Improvements:
- Refuses to recurse from filesystem roots (e.g. C:\) to avoid scanning system folders.
- Excludes paths that contain node_modules, \Windows\, Program Files, Program Files (x86), or AppData.
- Safer discovery of package.json / requirements.txt / Cargo.toml inside the provided RepoPath only.

Usage:
  powershell -ExecutionPolicy Bypass -File .\install-app.ps1 -RepoPath "C:\Users\Dave5\source\repos\tambourine-voice"
#>

param(
    [string]$RepoPath = (Get-Location).Path,
    [switch]$InstallDependencies = $false,
    [switch]$SkipChocolateyInstall = $false,
    [ValidateSet('npm','pnpm','yarn')][string]$NodePackageManager = 'npm',
    [string[]]$WhatToDo = @('node','python','rust')
)

function Write-ErrAndExit { param($msg,$code=1) Write-Error $msg; exit $code }

# refuse to run against root/system mounts
if ($RepoPath -match '^[A-Za-z]:\\$' -or $RepoPath -match '^[A-Za-z]:\\Users\\[^\\]+\\$') {
    Write-ErrAndExit "Refusing to recurse from a root or broad path ($RepoPath). Please point RepoPath to the repository folder (e.g. C:\\Users\\Dave5\\source\\repos\\tambourine-voice)."
}

if (-not (Test-Path $RepoPath)) {
    Write-ErrAndExit "Repo path does not exist: $RepoPath"
}

# helper to filter out unwanted paths
function Is-ExcludedPath {
    param($fullPath)
    $excludePatterns = @(
        '\\node_modules\\',
        '\\Windows\\',
        '\\winnt\\',
        '\\Program Files\\',
        '\\Program Files (x86)\\',
        '\\AppData\\',
        '\\$Recycle.Bin\\',
        '\\System Volume Information\\'
    )
    foreach ($p in $excludePatterns) {
        if ($fullPath -like "*$p*") { return $true }
    }
    return $false
}

function Find-Files {
    param($repoPath, $namePattern)
    # Use Get-ChildItem and filter out excluded directories in the pipeline
    Get-ChildItem -Path $repoPath -Filter $namePattern -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { -not (Is-ExcludedPath $_.FullName) }
}

function Run-Command {
    param($cmd,$args,$workDir)
    Write-Host "=> $cmd $args (in $workDir)"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $cmd
    $psi.Arguments = $args
    $psi.WorkingDirectory = $workDir
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError = $false
    $psi.UseShellExecute = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.WaitForExit()
    return $proc.ExitCode
}

# Begin main flow
Write-Host "RepoPath: $RepoPath"

# Node projects
if ($WhatToDo -contains 'node') {
    $pkgJsonFiles = Find-Files -repoPath $RepoPath -namePattern 'package.json'
    foreach ($pkg in $pkgJsonFiles) {
        $projDir = $pkg.DirectoryName
        if (Is-ExcludedPath $projDir) { continue }
        Write-Host "`n--- Node project: $projDir ---"
        # read package.json, but be tolerant
        try { $pjson = Get-Content -Raw -Path $pkg.FullName | ConvertFrom-Json } catch { $pjson = $null }
        $installCmd = switch ($NodePackageManager) { 'npm' { 'install' } default { 'install' } }
        if ($NodePackageManager -eq 'npm' -and (Test-Path (Join-Path $projDir 'package-lock.json'))) { $installCmd = 'ci' }

        $exit = Run-Command $NodePackageManager $installCmd $projDir
        if ($exit -ne 0) {
            Write-Warning "Node install ($NodePackageManager $installCmd) returned code $exit in $projDir. Continue to next project."
            continue
        }

        if ($pjson -and $pjson.scripts -and $pjson.scripts.build) {
            Write-Host "Found 'build' script. Running: $NodePackageManager run build"
            $exit = Run-Command $NodePackageManager 'run build' $projDir
            if ($exit -ne 0) { Write-Warning "Build script returned code $exit in $projDir." }
        } else {
            Write-Host "No build script found for $projDir."
        }
    }
}

# Python projects
if ($WhatToDo -contains 'python') {
    $reqFiles = Find-Files -repoPath $RepoPath -namePattern 'requirements.txt'
    foreach ($req in $reqFiles) {
        $projDir = $req.DirectoryName
        if (Is-ExcludedPath $projDir) { continue }
        Write-Host "`n--- Python project: $projDir ---"
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Write-Warning "python not found; skipping."; continue }
        $venvPath = Join-Path $projDir '.venv'
        if (-not (Test-Path $venvPath)) {
            $exit = Run-Command 'python' "-m venv `"$venvPath`"" $projDir
            if ($exit -ne 0) { Write-Warning "Failed to create venv in $projDir (exit $exit)."; continue }
        }
        $pipExe = Join-Path $venvPath 'Scripts\pip.exe'
        if (-not (Test-Path $pipExe)) { Write-Warning "pip not found in venv ($pipExe). Skipping pip install."; continue }
        Write-Host "Installing requirements from $($req.FullName)"
        $exit = Run-Command $pipExe "install -r `"$($req.FullName)`"" $projDir
        if ($exit -ne 0) { Write-Warning "pip install returned code $exit for $projDir." }
    }
}

# Rust projects
if ($WhatToDo -contains 'rust') {
    $cargoFiles = Find-Files -repoPath $RepoPath -namePattern 'Cargo.toml'
    foreach ($c in $cargoFiles) {
        $projDir = $c.DirectoryName
        if (Is-ExcludedPath $projDir) { continue }
        Write-Host "`n--- Rust project: $projDir ---"
        if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { Write-Warning "cargo not found; skipping."; continue }
        $exit = Run-Command 'cargo' 'build --release' $projDir
        if ($exit -ne 0) { Write-Warning "cargo build failed (exit $exit) in $projDir" } else { Write-Host "cargo build succeeded in $projDir" }
    }
}

Write-Host "`nAll done. If you still see system paths being scanned, make sure you supplied -RepoPath pointing to your repository directory (not C:\)."

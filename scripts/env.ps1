# Puts the Rust toolchain and the MinGW-w64 linker (WinLibs, installed with winget) on PATH for this PowerShell session.
# Usage:  . .\scripts\env.ps1      (note the leading dot)
$wl = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory -ErrorAction SilentlyContinue |
    Where-Object Name -like "BrechtSanders.WinLibs.POSIX.UCRT*" | Select-Object -First 1
if (-not $wl) { Write-Warning "WinLibs not found: install it with  winget install BrechtSanders.WinLibs.POSIX.UCRT" }
else { $env:PATH = "$($wl.FullName)\mingw64\bin;$env:PATH" }
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
rustc --version
gcc --version | Select-Object -First 1

# Runs the full benchmark and records the machine it ran on.
# Usage (from the rust\ folder, after  . .\scripts\env.ps1 ):
#     powershell -ExecutionPolicy Bypass -File scripts\run_benchmark.ps1 [-Big]
param([switch]$Big, [string]$Out = "results")

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$os = Get-CimInstance Win32_OperatingSystem
$ramGb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
$load = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
$info = @()
$info += "date:        $(Get-Date -Format s)"
$info += "cpu:         $($cpu.Name)  ($($cpu.NumberOfCores) cores, $($cpu.NumberOfLogicalProcessors) threads, max $($cpu.MaxClockSpeed) MHz)"
$info += "ram:         $ramGb GB"
$info += "os:          $($os.Caption) $($os.Version)"
$info += "rustc:       $(rustc --version)"
$info += "cpu load before the run (%): $load"
$info | Set-Content -Encoding utf8 (Join-Path $Out "machine.txt")

cargo build --release
$exe = Join-Path (Get-Location) "target\release\bench.exe"
$argsList = @("--out", $Out)
if ($Big) { $argsList += "--big" }
$log = Join-Path $Out "bench_log.txt"

# one logical CPU, above-normal priority: fewer migrations and less interference from other programs
$p = Start-Process -FilePath $exe -ArgumentList $argsList -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError (Join-Path $Out "bench_progress.txt")
$null = $p.Handle   # keep the handle so that ExitCode is available after the process ends
try { $p.ProcessorAffinity = [IntPtr]4; $p.PriorityClass = "AboveNormal" } catch { Write-Warning "could not set affinity/priority: $_" }
$p.WaitForExit()
Add-Content -Encoding utf8 (Join-Path $Out "machine.txt") "exit code: $($p.ExitCode)"

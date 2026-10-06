# Train runs (default: all 9 = 3 models x 3 seeds) one after another on the single GPU, on the
# de-duplicated split (src.data.dedup_split). Before each step it waits until no other
# src.train / src.predict process runs and enough commit memory is free.
#   -Only "cnn_scratch:123,resnet18:42"   train just these model:seed pairs
#   -LogSuffix "_rerun"                    log to runs\logs\{model}_s{seed}{suffix}.log, keeping older logs
#   -AfterPid 1234                         start only after this process (e.g. the eval step) has ended
#   -OverfitCheck                          run --overfit-batch once per model first (after a model change)
#   -PredictSplits "val"                   labeled prediction grids for these splits only
param([string]$Only = "", [string]$LogSuffix = "", [int]$AfterPid = 0, [switch]$OverfitCheck,
      [string]$PredictSplits = "test,val")
Set-Location "D:\Download\Data-20260919T021318Z-1-001 (2)\Data-20260919T021318Z-1-001\facial emotion"
$env:PYTHONUTF8 = "1"
$status = Join-Path (Get-Location) "runs\logs\queue_status.txt"
$minFreeGB = 4.5   # one training process needs ~4.3 GB of commit (WinError 1455 otherwise)

function Log($msg) {
    $line = "$(Get-Date -Format 'HH:mm') $msg`r`n"
    for ($i = 0; $i -lt 10; $i++) { try { [IO.File]::AppendAllText($status, $line); return } catch { Start-Sleep 2 } }
}
function Busy {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'src\.(train|predict|evaluate)' }).Count -gt 0
}
function FreeCommitGB { (Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB }
function WaitReady($what) {
    while (Busy) { Start-Sleep 60 }
    $warned = $false
    while ((FreeCommitGB) -lt $minFreeGB) {
        if (-not $warned) { Log ("WAITING MEMORY before {0}: {1:N1} GB commit free, need {2} GB" -f $what, (FreeCommitGB), $minFreeGB); $warned = $true }
        Start-Sleep 60
    }
    if ($warned) { Log "memory OK, continuing with $what" }
}

if ($AfterPid) { $p = Get-Process -Id $AfterPid -ErrorAction SilentlyContinue; if ($p) { $p.WaitForExit() } }
if ($Only) {
    $jobs = foreach ($j in $Only.Split(',')) { $m, $s = $j.Trim().Split(':'); @{ m = $m; s = [int]$s } }
} else {
    # seed by seed, so all three models have a result early
    $jobs = foreach ($s in 42, 123, 2024) { foreach ($m in "cnn_scratch", "resnet18", "mobilenet_v2") { @{ m = $m; s = $s } } }
}
Log "QUEUE start: $(@($jobs).Count) runs on the de-duplicated split$(if ($Only) { " ($Only)" })"
$overfitOk = @{}
foreach ($j in $jobs) {
    $m = $j.m; $s = $j.s; $log = "runs\logs\${m}_s$s$LogSuffix.log"
    $cfg = "configs/$m.yaml"

    if ($OverfitCheck -and -not $overfitOk.ContainsKey($m)) {
        WaitReady "overfit check $m"
        cmd /c "python -u -m src.train --config $cfg --seed $s --overfit-batch > runs\logs\queue_overfit.log 2>&1"
        $overfitOk[$m] = $LASTEXITCODE -eq 0
        Log ("OVERFIT {0} {1} (runs\logs\queue_overfit.log)" -f $(if ($overfitOk[$m]) { "PASS" } else { "FAILED" }), $m)
    }
    if ($OverfitCheck -and -not $overfitOk[$m]) { Log "SKIPPED $m s$s (overfit check failed)"; continue }

    WaitReady "dry-run $m s$s"
    cmd /c "python -u -m src.train --config $cfg --seed $s --dry-run > runs\logs\queue_dryrun.log 2>&1"
    if ($LASTEXITCODE -ne 0) { Log "DRYRUN FAILED $m s$s (runs\logs\queue_dryrun.log) - skipped"; continue }

    WaitReady "training $m s$s"
    Log "START $m s$s -> $log"
    cmd /c "python -u -m src.train --config $cfg --seed $s > $log 2>&1"
    $code = $LASTEXITCODE
    Log "END   $m s$s exit=$code"
    if ($code -ne 0) { continue }

    # run directory as printed by train.py ("run dir: runs\...")
    $hit = Select-String -Path $log -Pattern '^run dir: (.+)$' | Select-Object -First 1
    if (-not $hit) { Log "PREDICT SKIPPED $m s$s (run dir not found in $log)"; continue }
    $run = $hit.Matches[0].Groups[1].Value.Trim()
    foreach ($sp in $PredictSplits.Split(',')) {
        WaitReady "predict $sp $m s$s"
        cmd /c "python -u -m src.predict --run $run --split $sp >> $log 2>&1"
    }
    Log "PREDICT done $m s$s ($run)"
}
Log "ALL DONE"

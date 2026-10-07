# Rerun scheduler for pending experiments (2026-10-06), MMS excluded.
# Every experiment runs in its own Start-Process, serialized, logs to rerun_logs_20261006\
$py = "C:\anaconda_install\envs\pytorch\python.exe"
$repro = $PSScriptRoot
$logdir = Join-Path $repro "rerun_logs_20261006"
New-Item -ItemType Directory -Force $logdir | Out-Null

function Run-Exp {
    param([string]$Name, [string]$Script, [string]$Env = "")
    $out = Join-Path $logdir "$Name.out.log"; $err = Join-Path $logdir "$Name.err.log"
    Write-Host ("=== {0} START {1}" -f (Get-Date -Format 'HH:mm:ss'), $Name)
    if ($Env) { Invoke-Expression $Env }
    $p = Start-Process -FilePath $py -ArgumentList $Script -WorkingDirectory $repro `
        -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -NoNewWindow -Wait
    Write-Host ("=== {0} DONE  {1} exit={2}" -f (Get-Date -Format 'HH:mm:ss'), $Name, $p.ExitCode)
}

# 1 cost
Run-Exp "cost" "sec5_6_speedup\speedup_benchmark.py"
# 2 breadth x6
Run-Exp "breadth_riesz"     "sec5_9_breadth\riesz_periodic_check.py"
Run-Exp "breadth_rd"        "sec5_9_breadth\reaction_diffusion_check.py"
Run-Exp "breadth_sburgers"  "sec5_9_breadth\system_burgers_check.py"
Run-Exp "breadth_scoupled"  "sec5_9_breadth\system_coupled_check.py"
Run-Exp "breadth_ns"        "sec5_9_breadth\ns_vorticity_check.py"
Run-Exp "breadth_ns3d"      "sec5_9_breadth\ns3d_vorticity_check.py"
# 3 ablation
Run-Exp "ablation_b2" "sec5_8_ablation\b2_ablation.py"
Run-Exp "ablation_c1" "sec5_8_ablation\c_ablation.py"
# 4 Exp.3/4 with new unified 1-D table + Gamma fix
Run-Exp "exp34" "sec5_7_theory\theory_closure\theory_p15_p14.py"
# 5 B3 with PINO_LAM 0 and 0.5
Run-Exp "b3_lam0"  "sec5_4_B3_spacetime\b3_pino.py" '$env:PINO_LAM="0"'
Run-Exp "b3_lam05" "sec5_4_B3_spacetime\b3_pino.py" '$env:PINO_LAM="0.5"'
Write-Host "=== ALL DONE ==="

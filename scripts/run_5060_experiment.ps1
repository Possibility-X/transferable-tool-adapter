param(
    [Parameter(Mandatory=$true)]
    [string]$ExperimentId,

    [string]$Stage = "all"
)

$ErrorActionPreference = "Stop"
$env:UV_LINK_MODE = "copy"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

uv --cache-dir .uv-cache run --no-sync python scripts/verify_cuda_5060.py

uv --cache-dir .uv-cache run --no-sync python src/run_experiment.py --id $ExperimentId --stage $Stage

# Verify the documented Windows build from a new Git checkout and virtualenv.
param(
    [string]$PythonLauncher = "py"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunRoot = Join-Path $ProjectRoot ("runs\clean-install-" + [guid]::NewGuid().ToString("N"))
$Checkout = Join-Path $RunRoot "checkout"
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
$Started = Get-Date
$Status = "failed"
$Failure = $null
$Commit = (& git -C $ProjectRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "Cannot identify the source commit" }

function Invoke-Checked {
    param([string]$File, [string[]]$ArgumentList)
    & $File @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$File exited with code $LASTEXITCODE while running: $($ArgumentList -join ' ')"
    }
}

try {
    Invoke-Checked git @("clone", "--no-hardlinks", "--quiet", $ProjectRoot, $Checkout)
    Push-Location $Checkout
    try {
        Invoke-Checked $PythonLauncher @("-3.11", "-m", "venv", ".venv")
        $Python = Join-Path $Checkout ".venv\Scripts\python.exe"
        Invoke-Checked $Python @("-m", "pip", "install", "-r", "requirements-dev.txt")
        Invoke-Checked cargo @("build", "--locked")
        Invoke-Checked $Python @("tests\fixtures\create_eval_models.py")
        Invoke-Checked $Python @("tests\fixtures\create_pointwise_model.py")
        Invoke-Checked $Python @("scripts\doctor.py")
        Invoke-Checked cargo @("test", "--locked")
        Invoke-Checked $Python @("tests\fixtures\smoke_auto_raster.py")
        Invoke-Checked $Python @("tests\fixtures\smoke_job_memory.py")
        Invoke-Checked $Python @("tests\fixtures\smoke_whole_image_oracle.py")
        $Status = "passed"
    }
    finally {
        Pop-Location
    }
}
catch {
    $Failure = $_.Exception.Message
    throw
}
finally {
    $Record = [ordered]@{
        status = $Status
        source_commit = $Commit
        checkout = $Checkout
        started_utc = $Started.ToUniversalTime().ToString("o")
        finished_utc = (Get-Date).ToUniversalTime().ToString("o")
        error = $Failure
        checks = @("pinned Python packages", "locked Rust build", "installation doctor",
                   "Rust unit tests", "automatic raster schedules", "Windows Job Object cap",
                   "whole-image score oracle")
    }
    $Record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RunRoot "result.json")
    Write-Output "Clean-install record: $(Join-Path $RunRoot 'result.json')"
}

param(
    [string]$TargetRoot = "F:\research\重邮\汤佳\RAGraph-new"
)

$ErrorActionPreference = "Stop"

$PackageRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SourceRoot = Join-Path $PackageRoot "code_to_copy\RAGraph_graph_new"
$TargetGraphRoot = Join-Path $TargetRoot "RAGraph_graph_new"
$TargetUtilsRoot = Join-Path $TargetGraphRoot "ragraph_utils"

if (!(Test-Path -LiteralPath $TargetGraphRoot)) {
    throw "Target graph directory not found: $TargetGraphRoot"
}

if (!(Test-Path -LiteralPath $TargetUtilsRoot)) {
    throw "Target ragraph_utils directory not found: $TargetUtilsRoot"
}

Copy-Item -LiteralPath (Join-Path $SourceRoot "RAGraph_memory.py") `
    -Destination (Join-Path $TargetGraphRoot "RAGraph_memory.py") -Force

Copy-Item -LiteralPath (Join-Path $SourceRoot "ragraph_utils\DiffLiftRingSelector.py") `
    -Destination (Join-Path $TargetUtilsRoot "DiffLiftRingSelector.py") -Force

Copy-Item -LiteralPath (Join-Path $SourceRoot "ragraph_utils\TaskAwareRetriever.py") `
    -Destination (Join-Path $TargetUtilsRoot "TaskAwareRetriever.py") -Force

Copy-Item -LiteralPath (Join-Path $SourceRoot "ragraph_utils\ToyGraphBase.py") `
    -Destination (Join-Path $TargetUtilsRoot "ToyGraphBase.py") -Force

Copy-Item -LiteralPath (Join-Path $SourceRoot "ragraph_utils\__init__.py") `
    -Destination (Join-Path $TargetUtilsRoot "__init__.py") -Force

Write-Host "Installed ReTAG memory code to: $TargetGraphRoot"
Write-Host "Next: cd $TargetGraphRoot"
Write-Host "Run: python -m py_compile RAGraph_memory.py ragraph_utils\DiffLiftRingSelector.py ragraph_utils\TaskAwareRetriever.py ragraph_utils\ToyGraphBase.py"

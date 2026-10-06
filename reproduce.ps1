param(
    [string]$Python = 'python',
    [ValidateSet('review','full')][string]$Mode = 'review',
    [string]$ReferenceRoot,
    [string]$SourcePackage,
    [string]$SourceCache,
    [ValidateSet('exact','numerical')][string]$Profile,
    [string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
if ($Mode -eq 'review') {
    if (!$Profile -or !$OutputRoot) { throw 'Review mode requires an explicit Profile and a new OutputRoot.' }
    & $Python -B -X utf8 (Join-Path $PSScriptRoot 'code/reproduce_review.py') --root $PSScriptRoot --profile $Profile --output $OutputRoot
} else {
    if (!$ReferenceRoot -or !$SourcePackage -or !$SourceCache -or !$OutputRoot) {
        throw 'Full mode requires ReferenceRoot, SourcePackage, SourceCache and a new OutputRoot.'
    }
    & $Python -X utf8 (Join-Path $PSScriptRoot 'code/reproduce_full.py') --reference-root $ReferenceRoot --source-package $SourcePackage --source-cache $SourceCache --output-root $OutputRoot
}
if ($LASTEXITCODE -ne 0) { throw "Reproduction failed with exit code $LASTEXITCODE" }

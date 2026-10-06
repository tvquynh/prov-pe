param(
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$CandidateRoot = $PSScriptRoot
if (Test-Path -LiteralPath $OutputRoot) { throw 'OutputRoot already exists; use a fresh path.' }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$ResolvedOutput = (Resolve-Path -LiteralPath $OutputRoot).Path
& $Python (Join-Path $CandidateRoot 'code/reproduce_review.py') --root $CandidateRoot --output (Join-Path $ResolvedOutput 'primary') --profile numerical
if ($LASTEXITCODE -ne 0) { throw 'Original-artifact verification failed.' }
& $Python (Join-Path $CandidateRoot 'revision/code/reproduce_revision.py') --output (Join-Path $ResolvedOutput 'revision')
if ($LASTEXITCODE -ne 0) { throw 'Revision verification failed.' }
& $Python -O (Join-Path $CandidateRoot 'revision/code/test_revision_checks.py')
if ($LASTEXITCODE -ne 0) { throw 'Revision software tests failed.' }
Write-Output 'Both compact verification scopes and software tests passed. No training was run.'

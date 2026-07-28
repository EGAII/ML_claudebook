<#
validate.ps1 -- structural checks on the repo.

  * every .ipynb parses as JSON and has a valid nbformat 4 structure
  * every code cell is unexecuted (execution_count null, empty outputs)
  * every relative markdown link in docs/, README.md and the notebooks resolves to a real file

Does NOT check that the Python inside the notebooks runs -- that needs a Python
interpreter, which this machine doesn't have. Run the notebooks on Colab for that.

    powershell -ExecutionPolicy Bypass -File tools\validate.ps1
#>
[CmdletBinding()]
param([string]$Root = '')

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Root)) { $Root = Split-Path -Parent $PSScriptRoot }

$problems = 0
function Fail([string]$msg) { Write-Host "FAIL $msg" -ForegroundColor Red; $script:problems++ }
function Ok([string]$msg) { Write-Host "ok   $msg" -ForegroundColor DarkGray }

# ---------------------------------------------------------------- notebooks
Write-Host "`n=== notebook structure ===" -ForegroundColor Cyan
$books = Get-ChildItem -Path $Root -Filter '*.ipynb' -Recurse | Where-Object { $_.FullName -notlike '*ipynb_checkpoints*' }
foreach ($nb in $books) {
    $rel = $nb.FullName.Substring($Root.Length + 1)
    $text = [System.IO.File]::ReadAllText($nb.FullName)
    try { $j = ConvertFrom-Json $text } catch { Fail "$rel : invalid JSON -- $($_.Exception.Message)"; continue }

    if ($j.nbformat -ne 4) { Fail "$rel : nbformat is $($j.nbformat), expected 4"; continue }
    if (-not $j.cells -or $j.cells.Count -eq 0) { Fail "$rel : no cells"; continue }
    # Check the raw bytes: a culture-sensitive StartsWith("`u{FEFF}") matches every string,
    # because U+FEFF is an ignorable character in .NET string comparison.
    $head = [byte[]]::new(3)
    $fs = [System.IO.File]::OpenRead($nb.FullName)
    try { $null = $fs.Read($head, 0, 3) } finally { $fs.Dispose() }
    if ($head[0] -eq 0xEF -and $head[1] -eq 0xBB -and $head[2] -eq 0xBF) {
        Fail "$rel : file starts with a UTF-8 BOM"
    }

    $i = 0
    $bad = 0
    foreach ($c in $j.cells) {
        $i++
        if ($c.cell_type -notin @('code', 'markdown')) { Fail "$rel cell $i : cell_type '$($c.cell_type)'"; $bad++ }
        if ($null -eq $c.source) { Fail "$rel cell $i : no source"; $bad++ }
        if ($null -eq $c.metadata) { Fail "$rel cell $i : no metadata"; $bad++ }
        if ($c.cell_type -eq 'code') {
            if ($null -ne $c.execution_count) { Fail "$rel cell $i : execution_count should be null"; $bad++ }
            if ($null -eq $c.outputs) { Fail "$rel cell $i : code cell needs an outputs array"; $bad++ }
            elseif (@($c.outputs).Count -ne 0) { Fail "$rel cell $i : outputs should be empty"; $bad++ }
        }
    }
    if ($bad -eq 0) {
        $nCode = @($j.cells | Where-Object { $_.cell_type -eq 'code' }).Count
        $gpu = ''
        if ($j.metadata.accelerator -eq 'GPU') { $gpu = ' [gpu]' }
        Ok ("{0,-46} {1,3} cells, {2,2} code{3}" -f $rel, $j.cells.Count, $nCode, $gpu)
    }
}

# ---------------------------------------------------------------- links
Write-Host "`n=== relative links ===" -ForegroundColor Cyan
$linkRe = [regex]'\[[^\]]*\]\(([^)#:]+?)(?:#[^)]*)?\)'

function Test-Links([string]$sourcePath, [string]$content, [string]$baseDir) {
    $rel = $sourcePath.Substring($Root.Length + 1)
    $n = 0
    foreach ($m in $linkRe.Matches($content)) {
        $target = $m.Groups[1].Value.Trim()
        if ($target -match '^(https?|mailto)') { continue }
        if ($target -match '^[A-Za-z]:') { continue }
        if ($target -match '[<>*]') { continue }          # placeholder like <you>
        $resolved = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($baseDir, ($target -replace '/', '\')))
        if (-not (Test-Path $resolved)) { Fail "$rel -> $target (resolved: $resolved)" }
        else { $n++ }
    }
    return $n
}

$total = 0
foreach ($f in (Get-ChildItem -Path $Root -Filter '*.md' -Recurse | Where-Object { $_.FullName -notlike '*node_modules*' })) {
    $total += Test-Links $f.FullName ([System.IO.File]::ReadAllText($f.FullName)) $f.DirectoryName
}
foreach ($nb in $books) {
    $total += Test-Links $nb.FullName ([System.IO.File]::ReadAllText($nb.FullName)) $nb.DirectoryName
}
Write-Host "checked $total relative links"

# ---------------------------------------------------------------- expected layout
Write-Host "`n=== expected files ===" -ForegroundColor Cyan
$expected = @(
    'README.md', 'requirements.txt', '.gitignore', 'src/cv_utils.py',
    'tools/build_notebooks.ps1', 'tools/validate.ps1',
    'docs/00_setup_colab_vscode.md', 'docs/01_numpy_pandas.md', 'docs/02_ml_foundations.md',
    'docs/03_convolution.md', 'docs/04_cnn_classification.md', 'docs/05_transfer_learning.md',
    'docs/06_segmentation.md', 'docs/90_pytorch_cheatsheet.md', 'docs/91_debugging_playbook.md',
    'notebooks/01_numpy_pandas.ipynb', 'notebooks/02_linear_logistic_regression.ipynb',
    'notebooks/03_convolution_image_ops.ipynb', 'notebooks/04_cnn_classification.ipynb',
    'notebooks/05_transfer_learning.ipynb', 'notebooks/06_segmentation_unet.ipynb',
    'exercises/ex01_numpy_pandas.ipynb', 'exercises/ex02_regression.ipynb',
    'exercises/ex03_convolution.ipynb', 'exercises/ex04_cnn.ipynb',
    'exercises/ex05_transfer.ipynb', 'exercises/ex06_segmentation.ipynb',
    'exercises/solutions/sol01_numpy_pandas.ipynb', 'exercises/solutions/sol02_regression.ipynb',
    'exercises/solutions/sol03_convolution.ipynb', 'exercises/solutions/sol04_cnn.ipynb',
    'exercises/solutions/sol05_transfer.ipynb', 'exercises/solutions/sol06_segmentation.ipynb'
)
foreach ($e in $expected) {
    if (-not (Test-Path (Join-Path $Root $e))) { Fail "missing: $e" }
}
Write-Host "checked $($expected.Count) expected paths"

Write-Host ""
if ($problems -eq 0) {
    Write-Host "ALL CHECKS PASSED ($($books.Count) notebooks)" -ForegroundColor Green
} else {
    Write-Host "$problems problem(s)" -ForegroundColor Red
    exit 1
}

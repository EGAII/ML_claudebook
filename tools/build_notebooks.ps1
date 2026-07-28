<#
build_notebooks.ps1  --  turns plain-text lesson sources into Jupyter notebooks.

Why this exists
---------------
The .ipynb files in this repo are the deliverable, but raw notebook JSON is painful
to read and review in a diff. So every notebook is authored as a readable text file
under tools/nbsrc/ and compiled here. Works on Windows PowerShell 5.1 with no
Python / Node / nbformat installed.

Source format (tools/nbsrc/*.nbsrc)
-----------------------------------
    #out exercises/ex01_numpy_pandas.ipynb     <- required, first non-empty line
    #gpu on                                    <- optional, requests a GPU runtime in Colab
    #%% md
    ## A markdown cell
    #%% code
    print('a code cell')

Rules:
  * '#%% md' / '#%% code' start a new cell. Nothing else may start with '#%%'.
  * Header directives (#out, #gpu) must appear before the first '#%%'.
  * Blank lines at the start/end of a cell are trimmed.

Usage
-----
    powershell -ExecutionPolicy Bypass -File tools\build_notebooks.ps1
    powershell -ExecutionPolicy Bypass -File tools\build_notebooks.ps1 -Only 01
#>
[CmdletBinding()]
param(
    # Build only sources whose file name contains this substring.
    [string]$Only = '',
    # Repo root. Defaults to the parent of this script's folder.
    [string]$Root = ''
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = Split-Path -Parent $PSScriptRoot
}
$SrcDir = Join-Path $PSScriptRoot 'nbsrc'
if (-not (Test-Path $SrcDir)) { throw "source directory not found: $SrcDir" }

# --- JSON string escaping (hand-rolled: PS 5.1's ConvertTo-Json mangles < > & ') ---
function ConvertTo-JsonString([string]$s) {
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append('"')
    foreach ($ch in $s.ToCharArray()) {
        $code = [int]$ch
        if ($ch -eq '"') { [void]$sb.Append('\"') }
        elseif ($ch -eq '\') { [void]$sb.Append('\\') }
        elseif ($code -eq 8) { [void]$sb.Append('\b') }
        elseif ($code -eq 9) { [void]$sb.Append('\t') }
        elseif ($code -eq 10) { [void]$sb.Append('\n') }
        elseif ($code -eq 12) { [void]$sb.Append('\f') }
        elseif ($code -eq 13) { [void]$sb.Append('\r') }
        elseif ($code -lt 32) { [void]$sb.Append(('\u{0:x4}' -f $code)) }
        else { [void]$sb.Append($ch) }
    }
    [void]$sb.Append('"')
    return $sb.ToString()
}

# A cell's "source" is a JSON array of lines; every line but the last keeps its \n.
function ConvertTo-SourceArray([string[]]$lines) {
    if ($lines.Count -eq 0) { return '[]' }
    $parts = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $text = $lines[$i]
        if ($i -lt $lines.Count - 1) { $text = $text + "`n" }
        $parts.Add('    ' + (ConvertTo-JsonString $text))
    }
    return "[`n" + ($parts -join ",`n") + "`n   ]"
}

function New-Cell([string]$kind, [string[]]$lines) {
    $source = ConvertTo-SourceArray $lines
    if ($kind -eq 'md') {
        return "  {`n   `"cell_type`": `"markdown`",`n   `"metadata`": {},`n   `"source`": $source`n  }"
    }
    return "  {`n   `"cell_type`": `"code`",`n   `"execution_count`": null,`n   `"metadata`": {},`n   `"outputs`": [],`n   `"source`": $source`n  }"
}

function Get-TrimmedLines($lines) {
    $arr = @($lines)
    $start = 0
    $end = $arr.Count - 1
    while ($start -le $end -and [string]::IsNullOrWhiteSpace($arr[$start])) { $start++ }
    while ($end -ge $start -and [string]::IsNullOrWhiteSpace($arr[$end])) { $end-- }
    if ($end -lt $start) { return @() }
    # No comma-wrap here: the caller re-collects with @(), so returning the plain
    # slice keeps it flat. Wrapping would nest it and collapse the cell to one line.
    return $arr[$start..$end]
}

$sources = Get-ChildItem -Path $SrcDir -Filter '*.nbsrc' | Sort-Object Name
if ($Only -ne '') { $sources = $sources | Where-Object { $_.Name -like "*$Only*" } }
if (-not $sources) { throw "no .nbsrc files matched (-Only '$Only')" }

$built = 0
$failed = 0

foreach ($src in $sources) {
    $raw = [System.IO.File]::ReadAllText($src.FullName)
    $lines = $raw -split "`r`n|`n|`r"

    $outRel = $null
    $gpu = $false
    $cells = New-Object System.Collections.Generic.List[string]
    $current = New-Object System.Collections.Generic.List[string]

    function Flush-Cell {
        if ($null -ne $script:kind) {
            $body = @(Get-TrimmedLines $script:current)
            if ($body.Count -gt 0) { $script:cells.Add((New-Cell $script:kind $body)) }
        }
        $script:current = New-Object System.Collections.Generic.List[string]
    }
    # expose loop state to the helper above
    $script:kind = $null
    $script:current = $current
    $script:cells = $cells

    foreach ($line in $lines) {
        if ($line -match '^#%%\s*(md|markdown|code)\s*$') {
            Flush-Cell
            $tag = $Matches[1]
            if ($tag -eq 'code') { $script:kind = 'code' } else { $script:kind = 'md' }
            continue
        }
        if ($null -eq $script:kind) {
            # still in the header block
            if ($line -match '^#out\s+(.+?)\s*$') { $outRel = $Matches[1]; continue }
            if ($line -match '^#gpu\s+(on|off)\s*$') { $gpu = ($Matches[1] -eq 'on'); continue }
            continue
        }
        $script:current.Add($line)
    }
    Flush-Cell

    if (-not $outRel) { Write-Host "SKIP $($src.Name): missing '#out' directive" -ForegroundColor Yellow; $failed++; continue }
    if ($script:cells.Count -eq 0) { Write-Host "SKIP $($src.Name): no cells" -ForegroundColor Yellow; $failed++; continue }

    $meta = @(
        '  "kernelspec": {',
        '   "display_name": "Python 3",',
        '   "language": "python",',
        '   "name": "python3"',
        '  },',
        '  "language_info": {',
        '   "name": "python",',
        '   "version": "3.11"',
        '  },',
        '  "colab": {',
        '   "provenance": [],',
        '   "toc_visible": true',
        '  }'
    )
    if ($gpu) {
        $meta[$meta.Count - 1] = $meta[$meta.Count - 1] + ','
        $meta += '  "accelerator": "GPU"'
    }

    $json = @()
    $json += '{'
    $json += ' "cells": ['
    $json += ($script:cells -join ",`n")
    $json += ' ],'
    $json += ' "metadata": {'
    $json += $meta
    $json += ' },'
    $json += ' "nbformat": 4,'
    # nbformat_minor 4 (not 4.5) so per-cell "id" fields are not required.
    $json += ' "nbformat_minor": 4'
    $json += '}'
    $text = ($json -join "`n") + "`n"

    # Fail loudly rather than write a broken notebook.
    try { $null = ConvertFrom-Json $text }
    catch { Write-Host "FAIL $($src.Name): produced invalid JSON -- $($_.Exception.Message)" -ForegroundColor Red; $failed++; continue }

    $outPath = Join-Path $Root $outRel
    $outDir = Split-Path -Parent $outPath
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
    [System.IO.File]::WriteAllText($outPath, $text, (New-Object System.Text.UTF8Encoding($false)))

    $nCode = ([regex]::Matches($text, '"cell_type": "code"')).Count
    $nMd = ([regex]::Matches($text, '"cell_type": "markdown"')).Count
    $gpuTag = ''
    if ($gpu) { $gpuTag = ' [gpu]' }
    Write-Host ("OK   {0,-46} {1,3} cells ({2} md / {3} code){4}" -f $outRel, $script:cells.Count, $nMd, $nCode, $gpuTag) -ForegroundColor Green
    $built++
}

Write-Host ""
Write-Host "built $built notebook(s), $failed problem(s)"
if ($failed -gt 0) { exit 1 }

param(
    [int]$RowStart = 0,
    [int]$RowEnd = -1,
    [string]$SkipRows = "",
    [string]$PartNumbers = "2",
    [string]$Pos = "D441/03",
    [string]$PartNumberCsv = "data/part_number.csv",
    [switch]$NonMcqOnly,
    [switch]$CheckHuman,
    [switch]$IncludeUnmappedPartNumber
)

$pythonExe = "c:/Users/sran/Documents/data_analysis/.venv/Scripts/python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$args = @(
    "edit_distance.py",
    "--row-start", "$RowStart",
    "--row-end", "$RowEnd",
    "--part-number-csv", "$PartNumberCsv"
)

if ($SkipRows) {
    $args += @("--skip-rows", $SkipRows)
}
if ($PartNumbers) {
    $args += @("--part-numbers", $PartNumbers)
}
if ($Pos) {
    $args += @("--pos", $Pos)
}
if ($NonMcqOnly) {
    $args += "--non-mcq-only"
}
if ($CheckHuman) {
    $args += "--check-human"
}
if ($IncludeUnmappedPartNumber) {
    $args += "--include-unmapped-part-number"
}

& $pythonExe @args

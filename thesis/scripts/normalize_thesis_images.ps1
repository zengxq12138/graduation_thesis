param(
    [string]$InputPath = "D:\graduation_thesis\thesis\格式化第六版 - 格式规范版.docx",
    [string]$OutputPath = "",
    [double]$TargetWidth = 396.85
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-FullPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    $resolved = Resolve-Path -LiteralPath $PathValue -ErrorAction Stop
    return $resolved.ProviderPath
}

function Get-DefaultOutputPath {
    param([Parameter(Mandatory = $true)][string]$InputFile)
    $dir = Split-Path -Parent $InputFile
    $name = [System.IO.Path]::GetFileNameWithoutExtension($InputFile)
    $ext = [System.IO.Path]::GetExtension($InputFile)
    return [System.IO.Path]::Combine($dir, "$name - 图片统一$ext")
}

function Normalize-InlineShape {
    param(
        [Parameter(Mandatory = $true)]$InlineShape,
        [Parameter(Mandatory = $true)][double]$Width
    )
    try {
        $InlineShape.LockAspectRatio = -1
    } catch {
    }
    try {
        $InlineShape.Width = $Width
    } catch {
    }
}

function Normalize-Shape {
    param(
        [Parameter(Mandatory = $true)]$Shape,
        [Parameter(Mandatory = $true)][double]$Width
    )
    try {
        $Shape.LockAspectRatio = -1
    } catch {
    }
    try {
        $Shape.Width = $Width
    } catch {
    }
}

$inputFile = Resolve-FullPath -PathValue $InputPath

if (-not $OutputPath) {
    $OutputPath = Get-DefaultOutputPath -InputFile $inputFile
}

$outputFullPath = $OutputPath
if (Test-Path -LiteralPath $outputFullPath) {
    Remove-Item -LiteralPath $outputFullPath -Force
}
Copy-Item -LiteralPath $inputFile -Destination $outputFullPath
$outputFullPath = Resolve-FullPath -PathValue $outputFullPath

$word = $null
$document = $null
$inlineCount = 0
$shapeCount = 0

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $document = $word.Documents.Open($outputFullPath)

    foreach ($inlineShape in $document.InlineShapes) {
        Normalize-InlineShape -InlineShape $inlineShape -Width $TargetWidth
        $inlineCount++
    }

    foreach ($shape in $document.Shapes) {
        Normalize-Shape -Shape $shape -Width $TargetWidth
        $shapeCount++
    }

    $document.Repaginate() | Out-Null
    $document.Fields.Update() | Out-Null
    $document.Save()

    Write-Host "OutputPath: $outputFullPath"
    Write-Host "TargetWidth: $TargetWidth"
    Write-Host "InlineShapesUpdated: $inlineCount"
    Write-Host "ShapesUpdated: $shapeCount"
} finally {
    if ($document) {
        $document.Close()
    }
    if ($word) {
        $word.Quit()
    }
}

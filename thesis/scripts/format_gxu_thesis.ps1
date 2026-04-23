param(
    [string]$InputPath = "D:\graduation_thesis\thesis\格式化第六版 - 副本.docx",
    [string]$OutputPath = "",
    [switch]$ExportPdf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$wdOrganizerObjectStyles = 0
$wdStyleTypeParagraph = 1
$wdStyleTypeTable = 3
$wdAlignParagraphLeft = 0
$wdAlignParagraphCenter = 1
$wdLineSpaceSingle = 0
$wdLineSpaceExactly = 4
$wdLineStyleNone = 0
$wdLineStyleSingle = 1
$wdOutlineLevelBodyText = 10
$wdOutlineLevel1 = 1
$wdOutlineLevel2 = 2
$wdOutlineLevel3 = 3
$wdOutlineLevel4 = 4
$wdFooterPrimary = 1
$wdFooterFirstPage = 2
$wdFooterEvenPages = 3
$wdPageNumberStyleArabic = 0
$wdPageNumberStyleUppercaseRoman = 1
$wdAlignPageNumberCenter = 1
$wdMainTextStory = 1
$wdStatisticPages = 2
$wdWithInTable = 12

function Resolve-FullPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    $resolved = Resolve-Path -LiteralPath $PathValue -ErrorAction Stop
    return $resolved.ProviderPath
}

function Get-DefaultOutputPath {
    param([string]$InputFile)
    $dir = Split-Path -Parent $InputFile
    $name = [System.IO.Path]::GetFileNameWithoutExtension($InputFile)
    $ext = [System.IO.Path]::GetExtension($InputFile)
    return [System.IO.Path]::Combine($dir, "$name - 格式规范版$ext")
}

function Clean-ParagraphText {
    param($Text)
    if ($null -eq $Text) {
        return ""
    }
    return (($Text -replace "[`r`n`a`f`v]", " ") -replace "\s+", " ").Trim()
}

function Get-StyleByName {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string[]]$Names
    )
    foreach ($name in $Names) {
        try {
            return $Document.Styles.Item($name)
        } catch {
        }
    }
    throw "无法找到样式：$($Names -join ', ')"
}

function Ensure-ParagraphStyle {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Name,
        [string]$BaseStyle
    )
    try {
        return $Document.Styles.Item($Name)
    } catch {
        $style = $Document.Styles.Add($Name, $wdStyleTypeParagraph)
        if ($BaseStyle) {
            try {
                $style.BaseStyle = $Document.Styles.Item($BaseStyle)
            } catch {
            }
        }
        return $style
    }
}

function Copy-StyleSafe {
    param(
        [Parameter(Mandatory = $true)]$WordApp,
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath,
        [Parameter(Mandatory = $true)][string]$StyleName
    )
    try {
        $WordApp.OrganizerCopy($SourcePath, $DestinationPath, $StyleName, $wdOrganizerObjectStyles)
    } catch {
        Write-Host "跳过样式复制: $StyleName"
    }
}

function Set-FontSpec {
    param(
        [Parameter(Mandatory = $true)]$Font,
        [string]$FarEast,
        [string]$Ascii,
        [double]$Size,
        [int]$Bold = 0
    )
    if ($FarEast) {
        $Font.NameFarEast = $FarEast
    }
    if ($Ascii) {
        $Font.NameAscii = $Ascii
        $Font.NameOther = $Ascii
        $Font.Name = $Ascii
    }
    $Font.Size = $Size
    $Font.Bold = $Bold
}

function Set-ParagraphSpec {
    param(
        [Parameter(Mandatory = $true)]$ParagraphFormat,
        [int]$Alignment = $wdAlignParagraphLeft,
        [int]$LineSpacingRule = $wdLineSpaceExactly,
        [double]$LineSpacing = 20,
        [double]$SpaceBefore = 0,
        [double]$SpaceAfter = 0,
        [double]$CharacterUnitLeftIndent = 0,
        [double]$CharacterUnitFirstLineIndent = 0,
        [int]$OutlineLevel = $wdOutlineLevelBodyText
    )
    $ParagraphFormat.Alignment = $Alignment
    $ParagraphFormat.LineSpacingRule = $LineSpacingRule
    $ParagraphFormat.LineSpacing = $LineSpacing
    $ParagraphFormat.SpaceBefore = $SpaceBefore
    $ParagraphFormat.SpaceAfter = $SpaceAfter
    try {
        $ParagraphFormat.CharacterUnitLeftIndent = $CharacterUnitLeftIndent
    } catch {
    }
    try {
        $ParagraphFormat.CharacterUnitFirstLineIndent = $CharacterUnitFirstLineIndent
    } catch {
    }
    $ParagraphFormat.LeftIndent = 0
    $ParagraphFormat.FirstLineIndent = 0
    $ParagraphFormat.OutlineLevel = $OutlineLevel
}

function Configure-Style {
    param(
        [Parameter(Mandatory = $true)]$Style,
        [string]$FarEast,
        [string]$Ascii,
        [double]$Size,
        [int]$Bold = 0,
        [int]$Alignment = $wdAlignParagraphLeft,
        [double]$SpaceBefore = 0,
        [double]$SpaceAfter = 0,
        [double]$CharacterUnitLeftIndent = 0,
        [double]$CharacterUnitFirstLineIndent = 0,
        [int]$OutlineLevel = $wdOutlineLevelBodyText
    )
    Set-FontSpec -Font $Style.Font -FarEast $FarEast -Ascii $Ascii -Size $Size -Bold $Bold
    Set-ParagraphSpec -ParagraphFormat $Style.ParagraphFormat `
        -Alignment $Alignment `
        -LineSpacingRule $wdLineSpaceExactly `
        -LineSpacing 20 `
        -SpaceBefore $SpaceBefore `
        -SpaceAfter $SpaceAfter `
        -CharacterUnitLeftIndent $CharacterUnitLeftIndent `
        -CharacterUnitFirstLineIndent $CharacterUnitFirstLineIndent `
        -OutlineLevel $OutlineLevel
}

function Reset-ParagraphToStyle {
    param([Parameter(Mandatory = $true)]$Paragraph)
    try {
        $Paragraph.Range.Font.Reset()
    } catch {
    }
    try {
        $Paragraph.Range.ParagraphFormat.Reset()
    } catch {
    }
}

function Set-ParagraphStyle {
    param(
        [Parameter(Mandatory = $true)]$Paragraph,
        [Parameter(Mandatory = $true)]$Style
    )
    $Paragraph.Range.Style = $Style
    Reset-ParagraphToStyle -Paragraph $Paragraph
}

function Emphasize-KeywordPrefix {
    param(
        [Parameter(Mandatory = $true)]$Paragraph,
        [Parameter(Mandatory = $true)][string]$Prefix
    )
    $text = Clean-ParagraphText $Paragraph.Range.Text
    if (-not $text.StartsWith($Prefix)) {
        return
    }
    $prefixRange = $Paragraph.Range.Duplicate
    $prefixRange.End = $prefixRange.Start + $Prefix.Length
    $prefixRange.Font.Bold = 1
    $prefixRange.Font.NameFarEast = "黑体"
    $prefixRange.Font.NameAscii = "Times New Roman"
    $prefixRange.Font.NameOther = "Times New Roman"
}

function Is-MostlyAscii {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $false
    }
    $latin = ([regex]::Matches($Text, "[A-Za-z]")).Count
    $cjk = ([regex]::Matches($Text, "[\p{IsCJKUnifiedIdeographs}]")).Count
    return $latin -gt 0 -and $latin -ge ($cjk * 2)
}

function Is-FormulaParagraph {
    param(
        [Parameter(Mandatory = $true)]$Paragraph,
        [Parameter(Mandatory = $true)][string]$Text
    )
    if ($Text -match "^(式|图|表)\s*\d+") {
        return $false
    }
    if ($Paragraph.Range.OMaths.Count -gt 0) {
        return $true
    }
    if ($Text.Length -gt 120) {
        return $false
    }
    if ($Text -match "[=]" -and $Text -notmatch "[\p{IsCJKUnifiedIdeographs}]") {
        return $true
    }
    if ($Text -match "^[A-Za-z0-9\^\(\)\[\]\{\}\+\-\=\.,;:\/\\_ ]+$" -and $Text -match "[=]") {
        return $true
    }
    return $false
}

function Is-ListLikeParagraph {
    param([string]$Text)
    return $Text -match "^(?:\d+\.\s+|[（(]\d+[）)]|[一二三四五六七八九十]+、)"
}

function Is-FigureCaptionText {
    param([string]$Text)
    return $Text -match "^\s*图\s*\d+[-－—]\d+\s*\S" -and $Text.Length -le 80
}

function Is-TableCaptionText {
    param([string]$Text)
    return $Text -match "^\s*表\s*\d+[-－—]\d+\s*\S" -and $Text.Length -le 80
}

function Find-PreviousContentParagraphIndex {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][int]$StartIndex
    )
    for ($i = $StartIndex - 1; $i -ge 1; $i--) {
        $paragraph = $Document.Paragraphs.Item($i)
        $text = Clean-ParagraphText $paragraph.Range.Text
        if ($paragraph.Range.InlineShapes.Count -gt 0 -or $text) {
            return $i
        }
    }
    return $null
}

function Find-NextContentParagraphIndex {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][int]$StartIndex
    )
    for ($i = $StartIndex + 1; $i -le $Document.Paragraphs.Count; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        $text = Clean-ParagraphText $paragraph.Range.Text
        if ($paragraph.Range.InlineShapes.Count -gt 0 -or $text) {
            return $i
        }
    }
    return $null
}

function Normalize-EquationText {
    param([string]$Text)
    $normalized = Convert-LatexToPlainText -Text $Text
    $normalized = $normalized -replace "k\^\(\(l\)\)", "k^(l)"
    $normalized = $normalized -replace "k\^\(\(g\)\)", "k^(g)"
    $normalized = $normalized -replace "𝒟\s+\^", "𝒟^"
    $normalized = $normalized -replace "D\s+\^", "D^"
    $normalized = $normalized -replace "Recog\s+\(", "Recog("
    $normalized = $normalized -replace "Merge\s+\(", "Merge("
    $normalized = $normalized -replace "Prof\s+\(", "Prof("
    $normalized = $normalized -replace "\s+,", ","
    $normalized = $normalized -replace "\(\s+", "("
    $normalized = $normalized -replace "\s+\)", ")"
    return $normalized
}

function Convert-LatexToPlainText {
    param([string]$Text)
    $converted = $Text
    if ($null -eq $converted) {
        return ""
    }
    $converted = $converted -replace "\\\(", ""
    $converted = $converted -replace "\\\)", ""
    $converted = $converted -replace "\\hat\s*\{\s*\\mathcal\s*\{D\}\s*\}", "D^"
    $converted = $converted -replace "\\hat\s*\{\s*D\s*\}", "D^"
    $converted = $converted -replace "\\mathcal\s*\{D\}", "D"
    $converted = $converted -replace "\\mathcal\s*\{G\}", "G"
    $converted = $converted -replace "\\psi", "ψ"
    $converted = $converted -replace "\\phi", "ϕ"
    $converted = $converted -replace "\\Delta", "Δ"
    $converted = $converted -replace "\\cdot", "·"
    $converted = $converted -replace "\\mid", "|"
    $converted = $converted -replace "\bphi\s*\(", "ϕ("
    $converted = $converted -replace "\bpsi\s*\(", "ψ("
    $converted = $converted -replace "\bmathrm([A-Za-z]+)", '$1'
    $converted = $converted -replace "D┴\^", "D^"
    $converted = $converted -replace "\\,", " "
    $converted = $converted -replace "\\;", "; "
    $converted = $converted -replace "\\", ""
    $converted = $converted -replace "[{}]", ""
    return (Clean-ParagraphText $converted)
}

function Is-PureDisplayEquationText {
    param([string]$Text)
    if (-not $Text) {
        return $false
    }
    if ($Text.Length -gt 140) {
        return $false
    }
    $cjkCount = ([regex]::Matches($Text, "[\p{IsCJKUnifiedIdeographs}]")).Count
    if ($cjkCount -gt 4) {
        return $false
    }
    if ($Text -notmatch "[=\^_∈∧∨∘|∣ϕψ𝒟𝒱ℰΔ]") {
        return $false
    }
    return $true
}

function Ensure-TableStyle {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Name
    )
    try {
        return $Document.Styles.Item($Name)
    } catch {
        return $Document.Styles.Add($Name, $wdStyleTypeTable)
    }
}

function Normalize-ImageParagraph {
    param(
        [Parameter(Mandatory = $true)]$Paragraph,
        [double]$TargetWidth = 396.85
    )
    $paragraph.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
    $paragraph.Range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
    $paragraph.Range.ParagraphFormat.LineSpacing = 12
    $paragraph.Range.ParagraphFormat.SpaceBefore = 0
    $paragraph.Range.ParagraphFormat.SpaceAfter = 0
    $paragraph.Range.ParagraphFormat.LeftIndent = 0
    $paragraph.Range.ParagraphFormat.FirstLineIndent = 0
    try {
        $paragraph.Range.ParagraphFormat.CharacterUnitLeftIndent = 0
        $paragraph.Range.ParagraphFormat.CharacterUnitFirstLineIndent = 0
    } catch {
    }
    $paragraph.Range.ParagraphFormat.KeepWithNext = -1
    $paragraph.Range.ParagraphFormat.KeepTogether = -1
    foreach ($inlineShape in $paragraph.Range.InlineShapes) {
        try {
            $inlineShape.LockAspectRatio = -1
        } catch {
        }
        try {
            $inlineShape.Width = $TargetWidth
        } catch {
        }
    }
}

function Relocate-FloatingFigures {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)]$WordApp
    )
    $shapeEntries = @()
    foreach ($shape in $Document.Shapes) {
        try {
            if ($shape.Anchor.StoryType -ne $wdMainTextStory) {
                continue
            }
            $shapeEntries += [pscustomobject]@{
                Shape = $shape
                Page = $shape.Anchor.Information(3)
            }
        } catch {
        }
    }
    $shapeEntries = $shapeEntries | Sort-Object Page
    foreach ($entry in $shapeEntries) {
        $captionIndex = $null
        for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
            $paragraph = $Document.Paragraphs.Item($i)
            if ($paragraph.Range.StoryType -ne $wdMainTextStory) {
                continue
            }
            $text = Clean-ParagraphText $paragraph.Range.Text
            if (-not (Is-FigureCaptionText -Text $text)) {
                continue
            }
            $page = $paragraph.Range.Information(3)
            if ($page -lt $entry.Page) {
                continue
            }
            if ($page -gt ($entry.Page + 3)) {
                break
            }
            $prevIndex = Find-PreviousContentParagraphIndex -Document $Document -StartIndex $i
            if ($null -ne $prevIndex -and $Document.Paragraphs.Item($prevIndex).Range.InlineShapes.Count -gt 0) {
                continue
            }
            $captionIndex = $i
            break
        }
        if ($null -eq $captionIndex) {
            continue
        }
        try {
            $Document.Paragraphs.Item($captionIndex).Range.InsertParagraphBefore()
            $imageParagraph = $Document.Paragraphs.Item($captionIndex)
            try {
                $entry.Shape.Select()
                $WordApp.Selection.Copy()
                $imageParagraph.Range.Paste()
                $entry.Shape.Delete()
            } catch {
                try {
                    $inlineShape = $entry.Shape.ConvertToInlineShape()
                    $inlineShape.Range.Cut()
                    $imageParagraph.Range.Paste()
                } catch {
                }
            }
            Normalize-ImageParagraph -Paragraph $imageParagraph
        } catch {
        }
    }
}

function Normalize-FiguresAndCaptions {
    param([Parameter(Mandatory = $true)]$Document)
    $styleFigure = Get-StyleByName -Document $Document -Names @("图题")
    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        if ($paragraph.Range.StoryType -ne $wdMainTextStory) {
            continue
        }
        if ($paragraph.Range.InlineShapes.Count -gt 0) {
            Normalize-ImageParagraph -Paragraph $paragraph
            continue
        }
        $text = Clean-ParagraphText $paragraph.Range.Text
        if (-not (Is-FigureCaptionText -Text $text)) {
            continue
        }
        Set-ParagraphStyle -Paragraph $paragraph -Style $styleFigure
        $paragraph.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
        $paragraph.Range.ParagraphFormat.SpaceBefore = 0
        $paragraph.Range.ParagraphFormat.SpaceAfter = 0
        $paragraph.Range.ParagraphFormat.KeepTogether = -1
        $prevIndex = Find-PreviousContentParagraphIndex -Document $Document -StartIndex $i
        if ($null -ne $prevIndex -and $Document.Paragraphs.Item($prevIndex).Range.InlineShapes.Count -gt 0) {
            $Document.Paragraphs.Item($prevIndex).Range.ParagraphFormat.KeepWithNext = -1
        }
    }
}

function Normalize-DisplayEquations {
    param([Parameter(Mandatory = $true)]$Document)
    $styleFormula = Get-StyleByName -Document $Document -Names @("公式行")
    $styleBody = Get-StyleByName -Document $Document -Names @("正文")
    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        if ($paragraph.Range.StoryType -ne $wdMainTextStory) {
            continue
        }
        if ($paragraph.Range.Information($wdWithInTable)) {
            continue
        }
        if ($paragraph.Range.InlineShapes.Count -gt 0) {
            continue
        }
        $text = Clean-ParagraphText $paragraph.Range.Text
        if (-not $text) {
            continue
        }
        $styleName = ""
        try {
            $styleName = $paragraph.Style.NameLocal
        } catch {
        }
        if (Is-PureDisplayEquationText -Text $text) {
            $normalized = Normalize-EquationText -Text $text
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleFormula
            $equationRange = $paragraph.Range.Duplicate
            $equationRange.End = $equationRange.End - 1
            $equationRange.Text = $normalized
            try {
                [void]$Document.OMaths.Add($equationRange)
            } catch {
            }
            try {
                $paragraph.Range.OMaths.BuildUp()
            } catch {
            }
            $paragraph.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $paragraph.Range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            $paragraph.Range.ParagraphFormat.LineSpacing = 12
            $paragraph.Range.ParagraphFormat.SpaceBefore = 0
            $paragraph.Range.ParagraphFormat.SpaceAfter = 0
            $paragraph.Range.ParagraphFormat.KeepTogether = -1
            continue
        }
        if ($styleName -eq "公式行") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleBody
        }
    }
}

function Normalize-InlineMathMarkup {
    param([Parameter(Mandatory = $true)]$Document)
    $styleBody = Get-StyleByName -Document $Document -Names @("正文")
    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        if ($paragraph.Range.StoryType -ne $wdMainTextStory) {
            continue
        }
        if ($paragraph.Range.Information($wdWithInTable)) {
            continue
        }
        if ($paragraph.Range.InlineShapes.Count -gt 0) {
            continue
        }
        $rawText = $paragraph.Range.Text
        if ($null -eq $rawText) {
            continue
        }
        $needsCleanup = ($rawText.IndexOf('\') -ge 0) -or ($rawText -match "\bphi\s*\(") -or ($rawText -match "\bpsi\s*\(") -or ($rawText -match "\bmathrm[A-Za-z]+") -or ($rawText -match "D┴\^")
        if (-not $needsCleanup) {
            continue
        }
        $converted = Convert-LatexToPlainText -Text $rawText
        $current = Clean-ParagraphText $rawText
        if (-not $converted -or $converted -eq $current) {
            continue
        }
        $textRange = $paragraph.Range.Duplicate
        $textRange.End = $textRange.End - 1
        $textRange.Text = $converted
        try {
            $styleName = $paragraph.Style.NameLocal
            if ($styleName -ne "公式行") {
                Set-ParagraphStyle -Paragraph $paragraph -Style $styleBody
            }
        } catch {
        }
    }
}

function Normalize-References {
    param([Parameter(Mandatory = $true)]$Document)
    $styleTitle = Get-StyleByName -Document $Document -Names @("参考文献标题")
    $styleItem = Get-StyleByName -Document $Document -Names @("参考文献条目")
    $startIndex = $null
    $endIndex = $null
    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        $text = Clean-ParagraphText $Document.Paragraphs.Item($i).Range.Text
        if ($text -eq "参考文献") {
            $startIndex = $i
            break
        }
    }
    if ($null -eq $startIndex) {
        return
    }
    Set-ParagraphStyle -Paragraph $Document.Paragraphs.Item($startIndex) -Style $styleTitle
    for ($i = $startIndex + 1; $i -le $Document.Paragraphs.Count; $i++) {
        $text = Clean-ParagraphText $Document.Paragraphs.Item($i).Range.Text
        if ($text -eq "附录" -or $text -eq "致谢") {
            $endIndex = $i
            break
        }
    }
    if ($null -eq $endIndex) {
        $endIndex = $Document.Paragraphs.Count + 1
    }
    $entries = @()
    for ($i = $startIndex + 1; $i -lt $endIndex; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        $text = Clean-ParagraphText $paragraph.Range.Text
        if (-not $text) {
            continue
        }
        $entries += ($text -replace "^\[\d+\]\s*", "")
    }
    for ($i = $endIndex - 1; $i -gt $startIndex; $i--) {
        try {
            $Document.Paragraphs.Item($i).Range.Delete()
        } catch {
        }
    }
    if ($endIndex -le $Document.Paragraphs.Count) {
        $insertIndex = $startIndex + 1
        for ($n = $entries.Count; $n -ge 1; $n--) {
            $Document.Paragraphs.Item($insertIndex).Range.InsertParagraphBefore()
            $paragraph = $Document.Paragraphs.Item($insertIndex)
            $paragraph.Range.ListFormat.RemoveNumbers()
            $textRange = $paragraph.Range.Duplicate
            $textRange.End = $textRange.End - 1
            $textRange.Text = ("[{0}] {1}" -f $n, $entries[$n - 1])
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleItem
            $paragraph.Range.Font.Color = -16777216
            $paragraph.Range.Font.Underline = 0
            $paragraph.Range.ParagraphFormat.LeftIndent = 24
            $paragraph.Range.ParagraphFormat.FirstLineIndent = -24
            $paragraph.Range.ParagraphFormat.SpaceBefore = 0
            $paragraph.Range.ParagraphFormat.SpaceAfter = 0
            $paragraph.Range.ParagraphFormat.LineSpacingRule = $wdLineSpaceExactly
            $paragraph.Range.ParagraphFormat.LineSpacing = 20
        }
    }
}

function Remove-AllPageNumbers {
    param([Parameter(Mandatory = $true)]$Section)
    foreach ($footerType in @($wdFooterPrimary, $wdFooterFirstPage, $wdFooterEvenPages)) {
        try {
            $footer = $Section.Footers.Item($footerType)
            $footer.LinkToPrevious = $false
            while ($footer.PageNumbers.Count -gt 0) {
                $footer.PageNumbers.Item(1).Delete()
            }
        } catch {
        }
    }
}

function Ensure-PageNumbers {
    param(
        [Parameter(Mandatory = $true)]$Section,
        [Parameter(Mandatory = $true)][int]$NumberStyle,
        [Parameter(Mandatory = $true)][bool]$Restart,
        [int]$StartAt = 1,
        [bool]$IncludeFirstPage = $true
    )
    foreach ($footerType in @($wdFooterPrimary, $wdFooterFirstPage, $wdFooterEvenPages)) {
        try {
            $footer = $Section.Footers.Item($footerType)
            $footer.LinkToPrevious = $false
            while ($footer.PageNumbers.Count -gt 0) {
                $footer.PageNumbers.Item(1).Delete()
            }
            $footer.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            if ($footerType -ne $wdFooterFirstPage -or $IncludeFirstPage) {
                [void]$footer.PageNumbers.Add($wdAlignPageNumberCenter, $true)
            }
        } catch {
        }
    }
    $primary = $Section.Footers.Item($wdFooterPrimary)
    $primary.PageNumbers.NumberStyle = $NumberStyle
    $primary.PageNumbers.RestartNumberingAtSection = $Restart
    if ($Restart) {
        $primary.PageNumbers.StartingNumber = $StartAt
    }
}

function Get-SectionFirstText {
    param([Parameter(Mandatory = $true)]$Section)
    foreach ($paragraph in $Section.Range.Paragraphs) {
        $text = Clean-ParagraphText $paragraph.Range.Text
        if ($text) {
            return $text
        }
    }
    return ""
}

function Get-SectionIndexByFirstText {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$ExpectedText
    )
    for ($i = 1; $i -le $Document.Sections.Count; $i++) {
        if ((Get-SectionFirstText -Section $Document.Sections.Item($i)) -eq $ExpectedText) {
            return $i
        }
    }
    return $null
}

function Apply-ThesisStyles {
    param([Parameter(Mandatory = $true)]$Document)

    $styleHeading1 = Get-StyleByName -Document $Document -Names @("标题 1", "Heading 1")
    $styleHeading2 = Get-StyleByName -Document $Document -Names @("标题 2", "Heading 2")
    $styleHeading3 = Get-StyleByName -Document $Document -Names @("标题 3", "Heading 3")
    $styleHeading4 = Get-StyleByName -Document $Document -Names @("标题 4", "Heading 4")
    $styleBody = Get-StyleByName -Document $Document -Names @("正文")
    $styleBodyEnglish = Get-StyleByName -Document $Document -Names @("正文英文")
    $styleCnAbstractTitle = Get-StyleByName -Document $Document -Names @("中文摘要标题")
    $styleCnAbstractBody = Get-StyleByName -Document $Document -Names @("中文摘要正文")
    $styleKeyword = Get-StyleByName -Document $Document -Names @("中文关键词")
    $styleEnAbstractBody = Get-StyleByName -Document $Document -Names @("英文摘要正文")
    $styleTocTitle = Get-StyleByName -Document $Document -Names @("目录标题")
    $styleFigure = Get-StyleByName -Document $Document -Names @("图题")
    $styleTableCaption = Get-StyleByName -Document $Document -Names @("表题")
    $styleFormula = Get-StyleByName -Document $Document -Names @("公式行")
    $styleReferenceTitle = Get-StyleByName -Document $Document -Names @("参考文献标题")
    $styleReferenceItem = Get-StyleByName -Document $Document -Names @("参考文献条目")
    $styleToc1 = Get-StyleByName -Document $Document -Names @("TOC 1")
    $styleToc2 = Get-StyleByName -Document $Document -Names @("TOC 2")

    Configure-Style -Style $styleHeading1 -FarEast "黑体" -Ascii "Times New Roman" -Size 18 -Alignment $wdAlignParagraphCenter -OutlineLevel $wdOutlineLevel1
    Configure-Style -Style $styleHeading2 -FarEast "黑体" -Ascii "Times New Roman" -Size 15 -Alignment $wdAlignParagraphLeft -OutlineLevel $wdOutlineLevel2
    Configure-Style -Style $styleHeading3 -FarEast "黑体" -Ascii "Times New Roman" -Size 14 -Alignment $wdAlignParagraphLeft -OutlineLevel $wdOutlineLevel3
    Configure-Style -Style $styleHeading4 -FarEast "黑体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphLeft -OutlineLevel $wdOutlineLevel4

    Configure-Style -Style $styleBody -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphLeft -CharacterUnitFirstLineIndent 2
    Configure-Style -Style $styleBodyEnglish -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphLeft
    Configure-Style -Style $styleCnAbstractTitle -FarEast "黑体" -Ascii "Times New Roman" -Size 16 -Alignment $wdAlignParagraphCenter -OutlineLevel $wdOutlineLevel1
    Configure-Style -Style $styleCnAbstractBody -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphLeft -CharacterUnitFirstLineIndent 2
    Configure-Style -Style $styleKeyword -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphLeft -CharacterUnitLeftIndent 2
    Configure-Style -Style $styleEnAbstractBody -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphLeft -CharacterUnitFirstLineIndent 2
    Configure-Style -Style $styleTocTitle -FarEast "黑体" -Ascii "Times New Roman" -Size 16 -Alignment $wdAlignParagraphCenter -SpaceBefore 20 -SpaceAfter 20
    Configure-Style -Style $styleFigure -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphCenter
    Configure-Style -Style $styleTableCaption -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphCenter
    Configure-Style -Style $styleFormula -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphCenter
    Configure-Style -Style $styleReferenceTitle -FarEast "黑体" -Ascii "Times New Roman" -Size 18 -Alignment $wdAlignParagraphCenter -OutlineLevel $wdOutlineLevel1
    Configure-Style -Style $styleReferenceItem -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphLeft
    Configure-Style -Style $styleToc1 -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphLeft
    Configure-Style -Style $styleToc2 -FarEast "宋体" -Ascii "Times New Roman" -Size 12 -Alignment $wdAlignParagraphLeft
    try {
        $styleReferenceItem.ParagraphFormat.LeftIndent = 24
        $styleReferenceItem.ParagraphFormat.FirstLineIndent = -24
    } catch {
    }
    try {
        $styleToc2.ParagraphFormat.LeftIndent = 24
    } catch {
    }

    $chapterTitles = @(
        "绪论",
        "相关技术及开发环境介绍",
        "系统实现",
        "总结与展望",
        "附录",
        "致谢"
    )

    $state = "cover"
    $paragraphCount = $Document.Paragraphs.Count

    for ($i = 1; $i -le $paragraphCount; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        if ($paragraph.Range.StoryType -ne $wdMainTextStory) {
            continue
        }
        if ($paragraph.Range.Information($wdWithInTable)) {
            continue
        }

        $text = Clean-ParagraphText $paragraph.Range.Text
        if (-not $text) {
            continue
        }
        if ($paragraph.Range.InlineShapes.Count -gt 0) {
            continue
        }

        $styleName = ""
        try {
            $styleName = $paragraph.Style.NameLocal
        } catch {
        }

        if ($styleName -like "TOC *") {
            continue
        }
        if ($text -ne "目录" -and $paragraph.Range.Fields.Count -gt 0) {
            continue
        }

        if ($text -eq "摘要" -or $text -eq "ABSTRACT") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleCnAbstractTitle
            $state = if ($text -eq "摘要") { "cn_abstract" } else { "en_abstract" }
            continue
        }

        if ($text -eq "目录") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleTocTitle
            $state = "toc"
            continue
        }

        if ($text -eq "参考文献") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleReferenceTitle
            $state = "references"
            continue
        }

        if ($chapterTitles -contains $text) {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleHeading1
            $state = "body"
            continue
        }

        if ($text -match "^关键词[:：]") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleKeyword
            Emphasize-KeywordPrefix -Paragraph $paragraph -Prefix "关键词："
            $state = "after_cn_abstract"
            continue
        }

        if ($text -match "^Keywords[:：]") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleKeyword
            Emphasize-KeywordPrefix -Paragraph $paragraph -Prefix "Keywords:"
            $state = "after_en_abstract"
            continue
        }

        if ($state -eq "references") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleReferenceItem
            continue
        }

        if (Is-FigureCaptionText -Text $text) {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleFigure
            continue
        }

        if (Is-TableCaptionText -Text $text) {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleTableCaption
            continue
        }

        if ($text -match "^\s*\d+\.\d+\.\d+\.\d+") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleHeading4
            $state = "body"
            continue
        }

        if ($text -match "^\s*\d+\.\d+\.\d+") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleHeading3
            $state = "body"
            continue
        }

        if ($text -match "^\s*\d+\.\d+") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleHeading2
            $state = "body"
            continue
        }

        if ($styleName -eq "四级标题") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleHeading4
            $state = "body"
            continue
        }

        if ($state -eq "cn_abstract") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleCnAbstractBody
            continue
        }

        if ($state -eq "en_abstract") {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleEnAbstractBody
            continue
        }

        if ($state -eq "after_cn_abstract" -and (Is-MostlyAscii -Text $text)) {
            continue
        }

        if ($state -eq "toc") {
            continue
        }

        if ($state -eq "cover") {
            continue
        }

        if ((Is-FormulaParagraph -Paragraph $paragraph -Text $text)) {
            Set-ParagraphStyle -Paragraph $paragraph -Style $styleFormula
            continue
        }

        Set-ParagraphStyle -Paragraph $paragraph -Style $styleBody
        if ((Is-ListLikeParagraph -Text $text)) {
            try {
                $paragraph.Range.ParagraphFormat.CharacterUnitFirstLineIndent = 0
                $paragraph.Range.ParagraphFormat.CharacterUnitLeftIndent = 0
            } catch {
            }
            $paragraph.Range.ParagraphFormat.LeftIndent = 0
            $paragraph.Range.ParagraphFormat.FirstLineIndent = 0
        }
    }
}

function Apply-TableStyles {
    param([Parameter(Mandatory = $true)]$Document)
    foreach ($table in $Document.Tables) {
        try {
            $table.Style = "三线表"
        } catch {
        }
        $table.Rows.Alignment = 1
        $table.Borders.Item(1).LineStyle = $wdLineStyleSingle
        $table.Borders.Item(2).LineStyle = $wdLineStyleNone
        $table.Borders.Item(3).LineStyle = $wdLineStyleSingle
        $table.Borders.Item(4).LineStyle = $wdLineStyleNone
        $table.Borders.Item(5).LineStyle = $wdLineStyleNone
        $table.Borders.Item(6).LineStyle = $wdLineStyleNone
        foreach ($row in $table.Rows) {
            foreach ($cell in $row.Cells) {
                foreach ($paragraph in $cell.Range.Paragraphs) {
                    $text = Clean-ParagraphText $paragraph.Range.Text
                    if (-not $text) {
                        continue
                    }
                    try {
                        $paragraph.Range.Font.NameFarEast = "宋体"
                        $paragraph.Range.Font.NameAscii = "Times New Roman"
                        $paragraph.Range.Font.NameOther = "Times New Roman"
                        $paragraph.Range.Font.Size = 10.5
                        $paragraph.Range.Font.Bold = 0
                    } catch {
                    }
                    try {
                        $paragraph.Range.ParagraphFormat.LineSpacingRule = $wdLineSpaceExactly
                        $paragraph.Range.ParagraphFormat.LineSpacing = 18
                        $paragraph.Range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
                        $paragraph.Range.ParagraphFormat.LeftIndent = 0
                        $paragraph.Range.ParagraphFormat.FirstLineIndent = 0
                        $paragraph.Range.ParagraphFormat.CharacterUnitLeftIndent = 0
                        $paragraph.Range.ParagraphFormat.CharacterUnitFirstLineIndent = 0
                    } catch {
                    }
                }
            }
        }
        if ($table.Rows.Count -ge 1) {
            $table.Rows.Item(1).Borders.Item(3).LineStyle = $wdLineStyleSingle
            foreach ($cell in $table.Rows.Item(1).Cells) {
                try {
                    $cell.Range.Font.Bold = 1
                    $cell.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
                } catch {
                }
            }
        }
    }
}

function Apply-SectionLayout {
    param([Parameter(Mandatory = $true)]$Document)

    $sectionAbstract = Get-SectionIndexByFirstText -Document $Document -ExpectedText "摘 要"
    $sectionBody = Get-SectionIndexByFirstText -Document $Document -ExpectedText "绪论"

    if (-not $sectionAbstract) {
        $sectionAbstract = 2
    }
    if (-not $sectionBody) {
        $sectionBody = [Math]::Min($Document.Sections.Count, $sectionAbstract + 2)
    }

    for ($i = 1; $i -le $Document.Sections.Count; $i++) {
        $section = $Document.Sections.Item($i)
        $pageSetup = $section.PageSetup
        $pageSetup.TopMargin = 72
        $pageSetup.BottomMargin = 72
        $pageSetup.LeftMargin = 62.35
        $pageSetup.RightMargin = 62.35
        $pageSetup.HeaderDistance = 42.55
        $pageSetup.FooterDistance = 38.55

        if ($i -eq 1) {
            $pageSetup.DifferentFirstPageHeaderFooter = $true
            Ensure-PageNumbers -Section $section -NumberStyle $wdPageNumberStyleUppercaseRoman -Restart $true -StartAt 0 -IncludeFirstPage $false
            continue
        }

        if ($i -lt $sectionBody) {
            $pageSetup.DifferentFirstPageHeaderFooter = $false
            Ensure-PageNumbers -Section $section -NumberStyle $wdPageNumberStyleUppercaseRoman -Restart $false -StartAt 1
            continue
        }

        $pageSetup.DifferentFirstPageHeaderFooter = $false
        $restart = ($i -eq $sectionBody)
        Ensure-PageNumbers -Section $section -NumberStyle $wdPageNumberStyleArabic -Restart $restart -StartAt 1
    }
}

function Rebuild-Toc {
    param([Parameter(Mandatory = $true)]$Document)
    if ($Document.TablesOfContents.Count -eq 0) {
        return
    }
    $toc = $Document.TablesOfContents.Item(1)
    $range = $toc.Range
    $toc.Delete()
    [void]$Document.TablesOfContents.Add(
        $range,
        $true,
        1,
        2,
        $false,
        "",
        $true,
        $true,
        "中文摘要标题,1",
        $true,
        $false,
        $true
    )
}

$sourceTemplate = Resolve-FullPath -PathValue "D:\graduation_thesis\thesis\参考文档\广西大学毕业论文样式模板.docx"
$sourceTableTemplate = Resolve-FullPath -PathValue "D:\graduation_thesis\thesis\参考文档\优秀毕业论文模板.docx"
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

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $document = $word.Documents.Open($outputFullPath)
    try {
        if ($document.CompatibilityMode -gt 0) {
            $document.Convert() | Out-Null
        }
    } catch {
    }

    foreach ($styleName in @(
        "正文",
        "正文英文",
        "中文摘要标题",
        "中文摘要正文",
        "中文关键词",
        "英文摘要正文",
        "目录标题",
        "图题",
        "表题",
        "公式行",
        "参考文献标题",
        "参考文献条目",
        "TOC 1",
        "TOC 2",
        "TOC 3",
        "TOC 4"
    )) {
        Copy-StyleSafe -WordApp $word -SourcePath $sourceTemplate -DestinationPath $outputFullPath -StyleName $styleName
    }

    foreach ($styleName in @("三线表")) {
        Copy-StyleSafe -WordApp $word -SourcePath $sourceTableTemplate -DestinationPath $outputFullPath -StyleName $styleName
    }

    [void](Ensure-ParagraphStyle -Document $document -Name "正文" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "正文英文" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "中文摘要标题" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "中文摘要正文" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "中文关键词" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "英文摘要正文" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "目录标题" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "图题" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "表题" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "公式行" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "参考文献标题" -BaseStyle "Normal")
    [void](Ensure-ParagraphStyle -Document $document -Name "参考文献条目" -BaseStyle "Normal")
    [void](Ensure-TableStyle -Document $document -Name "三线表")

    Apply-ThesisStyles -Document $document
    Relocate-FloatingFigures -Document $document -WordApp $word
    Normalize-FiguresAndCaptions -Document $document
    Normalize-DisplayEquations -Document $document
    Normalize-InlineMathMarkup -Document $document
    Apply-TableStyles -Document $document
    Normalize-References -Document $document
    Apply-SectionLayout -Document $document
    Rebuild-Toc -Document $document

    $document.Repaginate() | Out-Null
    foreach ($toc in $document.TablesOfContents) {
        $toc.Update()
    }
    $document.Fields.Update() | Out-Null

    $document.Save()

    if ($ExportPdf) {
        $pdfPath = [System.IO.Path]::ChangeExtension($outputFullPath, ".pdf")
        $document.ExportAsFixedFormat($pdfPath, 17)
    }

    $pages = $document.ComputeStatistics($wdStatisticPages)
    Write-Host "OutputPath: $outputFullPath"
    Write-Host "PageCount: $pages"
    Write-Host "Sections: $($document.Sections.Count)"
    Write-Host "TablesOfContents: $($document.TablesOfContents.Count)"
    Write-Host "Tables: $($document.Tables.Count)"
} finally {
    if ($document) {
        $document.Close()
    }
    if ($word) {
        $word.Quit()
    }
}

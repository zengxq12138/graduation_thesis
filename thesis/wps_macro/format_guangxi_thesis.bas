Option Explicit

Private Type ThesisMarkers
    AbstractIdx As Long
    EnglishAbstractIdx As Long
    TocIdx As Long
    BodyIdx As Long
    ReferencesIdx As Long
    AppendixIdx As Long
    AckIdx As Long
End Type

Public Sub FormatGuangxiUniversityThesis()
    Dim doc As Document
    Dim marks As ThesisMarkers

    If Documents.Count = 0 Then
        MsgBox "请先在 WPS 中打开论文文档。", vbExclamation
        Exit Sub
    End If

    Set doc = ActiveDocument

    Application.ScreenUpdating = False
    On Error GoTo CleanFail

    RemoveLeadingRevisionMarker doc
    ApplyDocumentPageSetup doc

    marks = GetMarkers(doc)
    ValidateMarkers marks

    EnsureSectionsAndPageBreaks doc, marks
    ApplyDocumentPageSetup doc
    ConfigurePageNumbers doc

    marks = GetMarkers(doc)
    FormatParagraphsByRules doc, marks
    RebuildToc doc, marks
    FormatTocStyles doc

    doc.Repaginate
    doc.Fields.Update
    If doc.TablesOfContents.Count > 0 Then
        doc.TablesOfContents(1).Update
    End If

    Application.ScreenUpdating = True
    MsgBox "格式修正完成。" & vbCrLf & _
           "请重点人工复查：英文摘要题目、公式编号位置、表格内部字体、个别图表题注。", vbInformation
    Exit Sub

CleanFail:
    Application.ScreenUpdating = True
    MsgBox "宏执行失败：" & Err.Description, vbCritical
End Sub

Private Sub ValidateMarkers(ByVal marks As ThesisMarkers)
    If marks.AbstractIdx = 0 Then
        Err.Raise vbObjectError + 1000, , "未找到“摘 要/摘要”标题。"
    End If
    If marks.EnglishAbstractIdx = 0 Then
        Err.Raise vbObjectError + 1001, , "未找到“ABSTRACT”标题。"
    End If
    If marks.TocIdx = 0 Then
        Err.Raise vbObjectError + 1002, , "未找到“目 录/目录”标题。"
    End If
    If marks.BodyIdx = 0 Then
        Err.Raise vbObjectError + 1003, , "未找到正文起始章节（第一章）。"
    End If
End Sub

Private Sub RemoveLeadingRevisionMarker(ByVal doc As Document)
    Dim i As Long
    Dim text As String

    For i = 1 To doc.Paragraphs.Count
        text = CleanParagraphText(doc.Paragraphs(i).Range.Text)
        If text <> "" Then
            If IsRevisionMarker(text) Then
                doc.Paragraphs(i).Range.Delete
            End If
            Exit For
        End If
    Next i
End Sub

Private Function IsRevisionMarker(ByVal text As String) As Boolean
    IsRevisionMarker = RegexTest(text, "^第([一二三四五六七八九十百\d]+)版$")
End Function

Private Sub ApplyDocumentPageSetup(ByVal doc As Document)
    Dim sec As Section

    For Each sec In doc.Sections
        With sec.PageSetup
            .PaperSize = wdPaperA4
            .Orientation = wdOrientPortrait
            .TopMargin = CentimetersToPoints(2.54)
            .BottomMargin = CentimetersToPoints(2.54)
            .LeftMargin = CentimetersToPoints(2.2)
            .RightMargin = CentimetersToPoints(2.2)
            .HeaderDistance = CentimetersToPoints(1.5)
            .FooterDistance = CentimetersToPoints(1)
        End With
    Next sec
End Sub

Private Sub EnsureSectionsAndPageBreaks(ByVal doc As Document, ByVal marks As ThesisMarkers)
    If doc.Sections.Count = 1 Then
        InsertSectionBreakBefore doc.Paragraphs(marks.BodyIdx).Range
        InsertSectionBreakBefore doc.Paragraphs(marks.AbstractIdx).Range
    End If

    SetPageBreakBefore doc, marks.EnglishAbstractIdx, True
    SetPageBreakBefore doc, marks.TocIdx, True
    SetPageBreakBefore doc, marks.ReferencesIdx, True
    SetPageBreakBefore doc, marks.AppendixIdx, True
    SetPageBreakBefore doc, marks.AckIdx, True
End Sub

Private Sub InsertSectionBreakBefore(ByVal targetRange As Range)
    Dim rng As Range
    Set rng = ActiveDocument.Range(targetRange.Start, targetRange.Start)
    rng.InsertBreak wdSectionBreakNextPage
End Sub

Private Sub SetPageBreakBefore(ByVal doc As Document, ByVal paraIdx As Long, ByVal enabled As Boolean)
    If paraIdx <= 0 Or paraIdx > doc.Paragraphs.Count Then
        Exit Sub
    End If
    doc.Paragraphs(paraIdx).Range.ParagraphFormat.PageBreakBefore = enabled
End Sub

Private Sub ConfigurePageNumbers(ByVal doc As Document)
    Dim sec As Section
    Dim hdr As HeaderFooter
    Dim ftr As HeaderFooter

    For Each sec In doc.Sections
        sec.PageSetup.DifferentFirstPageHeaderFooter = False

        For Each hdr In sec.Headers
            hdr.LinkToPrevious = False
            hdr.Range.Text = ""
        Next hdr

        For Each ftr In sec.Footers
            ftr.LinkToPrevious = False
            ftr.Range.Text = ""
        Next ftr
    Next sec

    If doc.Sections.Count >= 2 Then
        With doc.Sections(2).Footers(wdHeaderFooterPrimary)
            .PageNumbers.RestartNumberingAtSection = True
            .PageNumbers.StartingNumber = 1
            .PageNumbers.NumberStyle = wdPageNumberStyleLowercaseRoman
            .PageNumbers.Add PageNumberAlignment:=wdAlignPageNumberCenter, FirstPage:=True
        End With
    End If

    If doc.Sections.Count >= 3 Then
        With doc.Sections(3).Footers(wdHeaderFooterPrimary)
            .PageNumbers.RestartNumberingAtSection = True
            .PageNumbers.StartingNumber = 1
            .PageNumbers.NumberStyle = wdPageNumberStyleArabic
            .PageNumbers.Add PageNumberAlignment:=wdAlignPageNumberCenter, FirstPage:=True
        End With
    End If
End Sub

Private Function GetMarkers(ByVal doc As Document) As ThesisMarkers
    Dim marks As ThesisMarkers
    Dim i As Long
    Dim text As String
    Dim normalized As String

    For i = 1 To doc.Paragraphs.Count
        text = CleanParagraphText(doc.Paragraphs(i).Range.Text)
        normalized = NormalizeText(text)
        If normalized = "" Then
            GoTo NextParagraph
        End If

        If marks.AbstractIdx = 0 Then
            If normalized = "摘要" Then
                marks.AbstractIdx = i
                GoTo NextParagraph
            End If
        End If

        If marks.EnglishAbstractIdx = 0 Then
            If UCase$(normalized) = "ABSTRACT" Then
                marks.EnglishAbstractIdx = i
                GoTo NextParagraph
            End If
        End If

        If marks.TocIdx = 0 Then
            If normalized = "目录" Then
                marks.TocIdx = i
                GoTo NextParagraph
            End If
        End If

        If marks.TocIdx > 0 And marks.BodyIdx = 0 Then
            If IsChapterHeading(text) Then
                marks.BodyIdx = i
                GoTo NextParagraph
            End If
        End If

        If marks.BodyIdx > 0 Then
            If marks.ReferencesIdx = 0 And normalized = "参考文献" Then
                marks.ReferencesIdx = i
                GoTo NextParagraph
            End If

            If marks.AppendixIdx = 0 And Left$(normalized, 2) = "附录" Then
                marks.AppendixIdx = i
                GoTo NextParagraph
            End If

            If marks.AckIdx = 0 And normalized = "致谢" Then
                marks.AckIdx = i
                GoTo NextParagraph
            End If
        End If

NextParagraph:
    Next i

    GetMarkers = marks
End Function

Private Sub FormatParagraphsByRules(ByVal doc As Document, ByVal marks As ThesisMarkers)
    Dim i As Long
    Dim p As Paragraph
    Dim text As String
    Dim coverSeq As Long

    For i = 1 To doc.Paragraphs.Count
        Set p = doc.Paragraphs(i)

        If p.Range.Information(wdWithInTable) Then
            GoTo NextParagraph
        End If

        text = CleanParagraphText(p.Range.Text)
        If text = "" Then
            GoTo NextParagraph
        End If

        If i < marks.AbstractIdx Then
            coverSeq = coverSeq + 1
            FormatCoverParagraph p, coverSeq, text
            GoTo NextParagraph
        End If

        If i = marks.AbstractIdx Then
            FormatChineseSpecialHeading p
            GoTo NextParagraph
        End If

        If i > marks.AbstractIdx And i < marks.EnglishAbstractIdx Then
            If IsEnglishTitleCandidate(doc, i, marks.EnglishAbstractIdx) Then
                FormatEnglishTitleParagraph p
            ElseIf IsChineseKeywordLine(text) Then
                FormatKeywordParagraph p, True
            Else
                FormatAbstractBodyParagraph p, True
            End If
            GoTo NextParagraph
        End If

        If i = marks.EnglishAbstractIdx Then
            FormatEnglishSpecialHeading p
            p.Range.ParagraphFormat.PageBreakBefore = True
            GoTo NextParagraph
        End If

        If i > marks.EnglishAbstractIdx And i < marks.TocIdx Then
            If IsEnglishKeywordLine(text) Then
                FormatKeywordParagraph p, False
            Else
                FormatAbstractBodyParagraph p, False
            End If
            GoTo NextParagraph
        End If

        If i = marks.TocIdx Then
            FormatChineseSpecialHeading p
            p.Range.ParagraphFormat.PageBreakBefore = True
            GoTo NextParagraph
        End If

        If i > marks.TocIdx And i < marks.BodyIdx Then
            GoTo NextParagraph
        End If

        If marks.ReferencesIdx > 0 And i = marks.ReferencesIdx Then
            FormatChineseSpecialHeading p
            p.Range.ParagraphFormat.PageBreakBefore = True
            GoTo NextParagraph
        End If

        If marks.AppendixIdx > 0 And i = marks.AppendixIdx Then
            FormatChineseSpecialHeading p
            p.Range.ParagraphFormat.PageBreakBefore = True
            GoTo NextParagraph
        End If

        If marks.AckIdx > 0 And i = marks.AckIdx Then
            FormatChineseSpecialHeading p
            p.Range.ParagraphFormat.PageBreakBefore = True
            GoTo NextParagraph
        End If

        If marks.ReferencesIdx > 0 And i > marks.ReferencesIdx Then
            If (marks.AppendixIdx = 0 Or i < marks.AppendixIdx) And (marks.AckIdx = 0 Or i < marks.AckIdx) Then
                FormatReferenceParagraph p
                GoTo NextParagraph
            End If
        End If

        If marks.AppendixIdx > 0 And i > marks.AppendixIdx Then
            If marks.AckIdx = 0 Or i < marks.AckIdx Then
                FormatBodyRegionParagraph doc, p, text, i, marks.BodyIdx
                GoTo NextParagraph
            End If
        End If

        If marks.AckIdx > 0 And i > marks.AckIdx Then
            FormatBodyParagraph p
            GoTo NextParagraph
        End If

        If i >= marks.BodyIdx Then
            If (marks.ReferencesIdx = 0 Or i < marks.ReferencesIdx) And _
               (marks.AppendixIdx = 0 Or i < marks.AppendixIdx) And _
               (marks.AckIdx = 0 Or i < marks.AckIdx) Then
                FormatBodyRegionParagraph doc, p, text, i, marks.BodyIdx
            End If
        End If

NextParagraph:
    Next i
End Sub

Private Sub FormatCoverParagraph(ByVal p As Paragraph, ByVal coverSeq As Long, ByVal text As String)
    Select Case coverSeq
        Case 1
            ApplyParagraphBase p, wdAlignParagraphCenter, 0, 0, 120, 18, 20
            ApplyMixedFont p.Range, "黑体", "Times New Roman", 36, False
            ApplyOutlineLevel p, wdOutlineLevelBodyText
        Case 2
            ApplyParagraphBase p, wdAlignParagraphCenter, 0, 0, 36, 24, 20
            ApplyMixedFont p.Range, "黑体", "Times New Roman", 26, False
            ApplyOutlineLevel p, wdOutlineLevelBodyText
        Case Else
            ApplyParagraphBase p, wdAlignParagraphCenter, 0, 0, 0, 6, 20
            ApplyMixedFont p.Range, "宋体", "Times New Roman", 14, False
            ApplyOutlineLevel p, wdOutlineLevelBodyText
            If InStr(text, "年") > 0 And InStr(text, "月") > 0 And InStr(text, "日") > 0 Then
                p.Range.ParagraphFormat.SpaceBefore = 72
                p.Range.ParagraphFormat.SpaceAfter = 0
            End If
    End Select
End Sub

Private Sub FormatChineseSpecialHeading(ByVal p As Paragraph)
    ApplyParagraphBase p, wdAlignParagraphCenter, 0, 0, 20, 20, 20
    ApplyMixedFont p.Range, "黑体", "Times New Roman", 16, False
    ApplyOutlineLevel p, wdOutlineLevel1
End Sub

Private Sub FormatEnglishSpecialHeading(ByVal p As Paragraph)
    ApplyParagraphBase p, wdAlignParagraphCenter, 0, 0, 20, 20, 20
    ApplyMixedFont p.Range, "Times New Roman", "Times New Roman", 16, True
    ApplyOutlineLevel p, wdOutlineLevel1
End Sub

Private Sub FormatEnglishTitleParagraph(ByVal p As Paragraph)
    ApplyParagraphBase p, wdAlignParagraphCenter, 0, 0, 0, 6, 20
    ApplyMixedFont p.Range, "Times New Roman", "Times New Roman", 16, False
    ApplyOutlineLevel p, wdOutlineLevelBodyText
End Sub

Private Sub FormatAbstractBodyParagraph(ByVal p As Paragraph, ByVal isChinese As Boolean)
    ApplyParagraphBase p, wdAlignParagraphJustify, 0.85, 0, 0, 0, 20
    If isChinese Then
        ApplyMixedFont p.Range, "宋体", "Times New Roman", 12, False
    Else
        ApplyMixedFont p.Range, "Times New Roman", "Times New Roman", 12, False
    End If
    ApplyOutlineLevel p, wdOutlineLevelBodyText
End Sub

Private Sub FormatKeywordParagraph(ByVal p As Paragraph, ByVal isChinese As Boolean)
    Dim labelEnd As Long
    Dim labelRange As Range
    Dim bodyRange As Range
    Dim text As String

    ApplyParagraphBase p, wdAlignParagraphLeft, 0, 0, 10, 0, 20
    text = CleanParagraphText(p.Range.Text)

    If isChinese Then
        ApplyMixedFont p.Range, "宋体", "Times New Roman", 12, False
    Else
        ApplyMixedFont p.Range, "Times New Roman", "Times New Roman", 12, False
    End If

    labelEnd = InStr(text, "：")
    If labelEnd = 0 Then
        labelEnd = InStr(text, ":")
    End If

    If labelEnd > 0 Then
        Set labelRange = p.Range.Duplicate
        labelRange.End = labelRange.Start + labelEnd
        If isChinese Then
            ApplyMixedFont labelRange, "黑体", "Times New Roman", 12, True
        Else
            ApplyMixedFont labelRange, "Times New Roman", "Times New Roman", 12, True
        End If

        Set bodyRange = p.Range.Duplicate
        bodyRange.Start = labelRange.End
        If isChinese Then
            ApplyMixedFont bodyRange, "宋体", "Times New Roman", 12, False
        Else
            ApplyMixedFont bodyRange, "Times New Roman", "Times New Roman", 12, False
        End If
    End If

    ApplyOutlineLevel p, wdOutlineLevelBodyText
End Sub

Private Sub FormatReferenceParagraph(ByVal p As Paragraph)
    ApplyParagraphBase p, wdAlignParagraphLeft, 0, 0, 0, 0, 20
    ApplyMixedFont p.Range, "宋体", "Times New Roman", 12, False
    ApplyOutlineLevel p, wdOutlineLevelBodyText
End Sub

Private Sub FormatBodyRegionParagraph(ByVal doc As Document, ByVal p As Paragraph, ByVal text As String, ByVal paraIdx As Long, ByVal bodyStartIdx As Long)
    If IsChapterHeading(text) Then
        TrySetBuiltInStyle p, wdStyleHeading1
        ApplyParagraphBase p, wdAlignParagraphCenter, 0, 0, 20, 20, 20
        ApplyMixedFont p.Range, "黑体", "Times New Roman", 18, False
        ApplyOutlineLevel p, wdOutlineLevel1
        p.Range.ParagraphFormat.KeepWithNext = True
        p.Range.ParagraphFormat.PageBreakBefore = (paraIdx <> bodyStartIdx)
        Exit Sub
    End If

    If IsFourthLevelHeading(text) Then
        TrySetBuiltInStyle p, wdStyleHeading4
        ApplyParagraphBase p, wdAlignParagraphLeft, 0, 0, 0, 0, 20
        ApplyMixedFont p.Range, "黑体", "Times New Roman", 12, False
        ApplyOutlineLevel p, wdOutlineLevel4
        p.Range.ParagraphFormat.KeepWithNext = True
        Exit Sub
    End If

    If IsThirdLevelHeading(text) Then
        TrySetBuiltInStyle p, wdStyleHeading3
        ApplyParagraphBase p, wdAlignParagraphLeft, 0, 0, 0, 0, 20
        ApplyMixedFont p.Range, "黑体", "Times New Roman", 14, False
        ApplyOutlineLevel p, wdOutlineLevel3
        p.Range.ParagraphFormat.KeepWithNext = True
        Exit Sub
    End If

    If IsSecondLevelHeading(text) Then
        TrySetBuiltInStyle p, wdStyleHeading2
        ApplyParagraphBase p, wdAlignParagraphLeft, 0, 0, 0, 0, 20
        ApplyMixedFont p.Range, "黑体", "Times New Roman", 15, False
        ApplyOutlineLevel p, wdOutlineLevel2
        p.Range.ParagraphFormat.KeepWithNext = True
        Exit Sub
    End If

    If IsFigureCaption(text) Then
        FormatCaptionParagraph p
        Exit Sub
    End If

    If IsTableCaption(text) Then
        FormatCaptionParagraph p
        Exit Sub
    End If

    If IsEquationNumber(text) Then
        ApplyParagraphBase p, wdAlignParagraphRight, 0, 0, 0, 0, 20
        ApplyMixedFont p.Range, "Times New Roman", "Times New Roman", 12, False
        ApplyOutlineLevel p, wdOutlineLevelBodyText
        Exit Sub
    End If

    FormatBodyParagraph p
End Sub

Private Sub FormatCaptionParagraph(ByVal p As Paragraph)
    ApplyParagraphBase p, wdAlignParagraphCenter, 0, 0, 6, 6, 20
    ApplyMixedFont p.Range, "楷体", "Times New Roman", 10.5, False
    ApplyOutlineLevel p, wdOutlineLevelBodyText
End Sub

Private Sub FormatBodyParagraph(ByVal p As Paragraph)
    ApplyParagraphBase p, wdAlignParagraphJustify, 0.85, 0, 0, 0, 20
    ApplyMixedFont p.Range, "宋体", "Times New Roman", 12, False
    ApplyOutlineLevel p, wdOutlineLevelBodyText
End Sub

Private Sub ApplyParagraphBase(ByVal p As Paragraph, ByVal alignment As WdParagraphAlignment, ByVal firstIndentCm As Double, ByVal leftIndentCm As Double, ByVal beforePt As Single, ByVal afterPt As Single, ByVal lineSpacingPt As Single)
    With p.Range.ParagraphFormat
        .Alignment = alignment
        .LeftIndent = CentimetersToPoints(leftIndentCm)
        .RightIndent = 0
        .FirstLineIndent = CentimetersToPoints(firstIndentCm)
        .SpaceBefore = beforePt
        .SpaceAfter = afterPt
        .LineSpacingRule = wdLineSpaceExactly
        .LineSpacing = lineSpacingPt
        .KeepWithNext = False
        .KeepTogether = False
        .WidowControl = False
    End With
End Sub

Private Sub ApplyMixedFont(ByVal rng As Range, ByVal eastAsiaFont As String, ByVal westernFont As String, ByVal sizePt As Single, ByVal isBold As Boolean)
    With rng.Font
        .Name = westernFont
        .NameAscii = westernFont
        .NameOther = westernFont
        .NameFarEast = eastAsiaFont
        .Size = sizePt
        .Bold = isBold
        .Italic = False
    End With
End Sub

Private Sub ApplyOutlineLevel(ByVal p As Paragraph, ByVal level As WdOutlineLevel)
    p.Range.ParagraphFormat.OutlineLevel = level
End Sub

Private Sub TrySetBuiltInStyle(ByVal p As Paragraph, ByVal builtInStyle As WdBuiltinStyle)
    On Error Resume Next
    p.Style = ActiveDocument.Styles(builtInStyle)
    On Error GoTo 0
End Sub

Private Sub RebuildToc(ByVal doc As Document, ByVal marks As ThesisMarkers)
    Dim deleteRange As Range
    Dim insertRange As Range

    If marks.TocIdx = 0 Or marks.BodyIdx = 0 Then
        Exit Sub
    End If

    Set deleteRange = doc.Range(doc.Paragraphs(marks.TocIdx).Range.End, doc.Paragraphs(marks.BodyIdx).Range.Start)
    If Len(NormalizeText(deleteRange.Text)) > 0 Then
        deleteRange.Delete
    End If

    Do While doc.TablesOfContents.Count > 0
        doc.TablesOfContents(1).Delete
    Loop

    Set insertRange = doc.Paragraphs(marks.TocIdx).Range.Duplicate
    insertRange.Collapse wdCollapseEnd
    insertRange.InsertParagraphAfter
    insertRange.Collapse wdCollapseEnd

    doc.TablesOfContents.Add _
        Range:=insertRange, _
        UseHeadingStyles:=True, _
        UpperHeadingLevel:=1, _
        LowerHeadingLevel:=2, _
        UseFields:=False, _
        RightAlignPageNumbers:=True, _
        IncludePageNumbers:=True, _
        UseHyperlinks:=False, _
        HidePageNumbersInWeb:=False, _
        UseOutlineLevels:=True
End Sub

Private Sub FormatTocStyles(ByVal doc As Document)
    On Error Resume Next

    ApplyStyleFormatting doc.Styles(wdStyleTOC1), "宋体", "Times New Roman", 12, False, 0
    ApplyStyleFormatting doc.Styles(wdStyleTOC2), "宋体", "Times New Roman", 12, False, 0.74

    On Error GoTo 0
End Sub

Private Sub ApplyStyleFormatting(ByVal styleObj As Style, ByVal eastAsiaFont As String, ByVal westernFont As String, ByVal sizePt As Single, ByVal isBold As Boolean, ByVal leftIndentCm As Double)
    With styleObj.Font
        .Name = westernFont
        .NameAscii = westernFont
        .NameOther = westernFont
        .NameFarEast = eastAsiaFont
        .Size = sizePt
        .Bold = isBold
    End With

    With styleObj.ParagraphFormat
        .FirstLineIndent = 0
        .LeftIndent = CentimetersToPoints(leftIndentCm)
        .RightIndent = 0
        .LineSpacingRule = wdLineSpaceExactly
        .LineSpacing = 20
        .SpaceBefore = 0
        .SpaceAfter = 0
    End With
End Sub

Private Function IsEnglishTitleCandidate(ByVal doc As Document, ByVal idx As Long, ByVal englishAbstractIdx As Long) As Boolean
    Dim prevIdx As Long
    Dim text As String

    prevIdx = PreviousNonEmptyParagraphIndex(doc, englishAbstractIdx)
    If idx <> prevIdx Then
        Exit Function
    End If

    text = CleanParagraphText(doc.Paragraphs(idx).Range.Text)
    If text = "" Then
        Exit Function
    End If

    If IsChineseKeywordLine(text) Or IsEnglishKeywordLine(text) Then
        Exit Function
    End If

    If HasChineseCharacters(text) Then
        Exit Function
    End If

    If Len(text) < 10 Then
        Exit Function
    End If

    IsEnglishTitleCandidate = True
End Function

Private Function PreviousNonEmptyParagraphIndex(ByVal doc As Document, ByVal idx As Long) As Long
    Dim i As Long
    For i = idx - 1 To 1 Step -1
        If CleanParagraphText(doc.Paragraphs(i).Range.Text) <> "" Then
            PreviousNonEmptyParagraphIndex = i
            Exit Function
        End If
    Next i
End Function

Private Function HasChineseCharacters(ByVal text As String) As Boolean
    Dim i As Long
    Dim ch As String
    Dim code As Long

    For i = 1 To Len(text)
        ch = Mid$(text, i, 1)
        code = AscW(ch)
        If code < 0 Then
            code = code + 65536
        End If
        If code >= 19968 And code <= 40959 Then
            HasChineseCharacters = True
            Exit Function
        End If
    Next i
End Function

Private Function IsChineseKeywordLine(ByVal text As String) As Boolean
    Dim normalized As String
    normalized = NormalizeText(text)
    IsChineseKeywordLine = (Left$(normalized, 3) = "关键词")
End Function

Private Function IsEnglishKeywordLine(ByVal text As String) As Boolean
    Dim normalized As String
    normalized = UCase$(NormalizeText(text))
    IsEnglishKeywordLine = (Left$(normalized, 9) = "KEYWORDS:")
End Function

Private Function IsChapterHeading(ByVal text As String) As Boolean
    IsChapterHeading = RegexTest(text, "^第[一二三四五六七八九十百]+章[\s　]+.+$")
End Function

Private Function IsSecondLevelHeading(ByVal text As String) As Boolean
    IsSecondLevelHeading = RegexTest(text, "^\d+\.\d+[\s　]+.+$")
End Function

Private Function IsThirdLevelHeading(ByVal text As String) As Boolean
    IsThirdLevelHeading = RegexTest(text, "^\d+\.\d+\.\d+[\s　]+.+$")
End Function

Private Function IsFourthLevelHeading(ByVal text As String) As Boolean
    IsFourthLevelHeading = RegexTest(text, "^\d+\.\d+\.\d+\.\d+[\s　]+.+$")
End Function

Private Function IsFigureCaption(ByVal text As String) As Boolean
    IsFigureCaption = RegexTest(text, "^图\d+-\d+.+$") And Len(text) <= 60
End Function

Private Function IsTableCaption(ByVal text As String) As Boolean
    If Not RegexTest(text, "^表\d+-\d+.+$") Then
        Exit Function
    End If
    If Len(text) > 60 Then
        Exit Function
    End If
    If Right$(text, 1) = "。" Then
        Exit Function
    End If
    If InStr(text, "给出") > 0 Or InStr(text, "汇总") > 0 Or InStr(text, "列出") > 0 Or _
       InStr(text, "展示") > 0 Or InStr(text, "如下") > 0 Or InStr(text, "所示") > 0 Or _
       InStr(text, "说明") > 0 Then
        Exit Function
    End If
    IsTableCaption = True
End Function

Private Function IsEquationNumber(ByVal text As String) As Boolean
    IsEquationNumber = RegexTest(text, "^[\(（]?\d+-\d+[\)）]?$")
End Function

Private Function RegexTest(ByVal text As String, ByVal pattern As String) As Boolean
    Dim re As Object
    Set re = CreateObject("VBScript.RegExp")
    re.Global = False
    re.IgnoreCase = False
    re.MultiLine = False
    re.Pattern = pattern
    RegexTest = re.Test(text)
End Function

Private Function CleanParagraphText(ByVal text As String) As String
    text = Replace(text, vbCr, "")
    text = Replace(text, Chr(11), "")
    text = Replace(text, Chr(7), "")
    CleanParagraphText = Trim$(text)
End Function

Private Function NormalizeText(ByVal text As String) As String
    text = CleanParagraphText(text)
    text = Replace(text, " ", "")
    text = Replace(text, vbTab, "")
    text = Replace(text, ChrW(12288), "")
    NormalizeText = text
End Function

---
name: docx
description: "Use this skill for creating, reading, editing, or manipulating Word documents (.docx files). Triggers include: any mention of 'Word doc', '.docx', or requests for professional documents with formatting like tables of contents, headings, page numbers. Also use when extracting or reorganizing .docx content, performing find-and-replace, or converting content into a polished Word document. Do NOT use for PDFs, spreadsheets, or general coding."
license: CC-BY-SA-4.0
---

# DOCX creation, editing, and analysis

## Overview

A .docx file is a ZIP archive containing XML files.

## Quick Reference

| Task | Approach |
|------|----------|
| Read/analyze content | `pandoc` or unpack for raw XML |
| Create new document | Use `docx-js` via Node.js |
| Edit existing document | Unpack -> edit XML -> repack |

### Converting .doc to .docx

Legacy `.doc` files must be converted before editing:

```bash
libreoffice --headless --convert-to docx document.doc
```

### Reading Content

```bash
# Text extraction with tracked changes
pandoc --track-changes=all document.docx -o output.md

# Raw XML access - unzip and inspect
mkdir -p unpacked && unzip -o document.docx -d unpacked/
```

### Converting to Images

```bash
libreoffice --headless --convert-to pdf document.docx
pdftoppm -jpeg -r 150 document.pdf page
```

---

## Creating New Documents

Generate .docx files with JavaScript using the `docx` npm package, then validate.

### Setup
```javascript
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        Header, Footer, AlignmentType, PageOrientation, LevelFormat, ExternalHyperlink,
        InternalHyperlink, Bookmark, FootnoteReferenceRun,
        TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
        VerticalAlign, PageNumber, PageBreak } = require('docx');

const doc = new Document({ sections: [{ children: [/* content */] }] });
Packer.toBuffer(doc).then(buffer => fs.writeFileSync("doc.docx", buffer));
```

Run via `bash`:
```bash
npm install -g docx
node generate-doc.js
```

### Page Size

```javascript
// docx-js defaults to A4, not US Letter - always set explicitly
sections: [{
  properties: {
    page: {
      size: {
        width: 12240,   // 8.5 inches in DXA
        height: 15840   // 11 inches in DXA
      },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
    }
  },
  children: [/* content */]
}]
```

| Paper | Width | Height | Content Width (1" margins) |
|-------|-------|--------|---------------------------|
| US Letter | 12,240 | 15,840 | 9,360 |
| A4 (default) | 11,906 | 16,838 | 9,026 |

### Styles (Override Built-in Headings)

Use Arial as the default font (universally supported).

```javascript
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 180, after: 180 }, outlineLevel: 1 } },
    ]
  },
  sections: [{ children: [
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Title")] }),
  ]}]
});
```

### Lists (NEVER use unicode bullets)

```javascript
// CORRECT - use numbering config with LevelFormat.BULLET
const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{ children: [
    new Paragraph({ numbering: { reference: "bullets", level: 0 },
      children: [new TextRun("Bullet item")] }),
  ]}]
});
```

### Tables

**Tables need dual widths** - set both `columnWidths` on the table AND `width` on each cell.

```javascript
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [4680, 4680],
  rows: [
    new TableRow({
      children: [
        new TableCell({
          borders,
          width: { size: 4680, type: WidthType.DXA },
          shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun("Cell")] })]
        })
      ]
    })
  ]
})
```

Always use `WidthType.DXA` - never `WidthType.PERCENTAGE`.

### Images

```javascript
new Paragraph({
  children: [new ImageRun({
    type: "png",
    data: fs.readFileSync("image.png"),
    transformation: { width: 200, height: 150 },
    altText: { title: "Title", description: "Desc", name: "Name" }
  })]
})
```

### Critical Rules for docx-js

- **Set page size explicitly** - defaults to A4
- **Never use `\n`** - use separate Paragraph elements
- **Never use unicode bullets** - use `LevelFormat.BULLET`
- **PageBreak must be in Paragraph**
- **ImageRun requires `type`** - always specify png/jpg/etc
- **Tables need dual widths** - `columnWidths` + cell `width`
- **Use `ShadingType.CLEAR`** - never SOLID for table shading
- **TOC requires HeadingLevel only** - no custom styles

---

## Editing Existing Documents

### Step 1: Unpack
```bash
mkdir -p unpacked && unzip -o document.docx -d unpacked/
# Pretty-print for editing
xmllint --format unpacked/word/document.xml > unpacked/word/document_fmt.xml
mv unpacked/word/document_fmt.xml unpacked/word/document.xml
```

### Step 2: Edit XML

Edit files in `unpacked/word/` using `file_edit` for string replacement.

**Use "Prometheus" as the author** for tracked changes and comments, unless the user requests a different name.

**Tracked Changes:**
```xml
<w:ins w:id="1" w:author="Prometheus" w:date="2026-01-01T00:00:00Z">
  <w:r><w:t>inserted text</w:t></w:r>
</w:ins>
```

```xml
<w:del w:id="2" w:author="Prometheus" w:date="2026-01-01T00:00:00Z">
  <w:r><w:delText>deleted text</w:delText></w:r>
</w:del>
```

### Step 3: Pack
```bash
cd unpacked && zip -r ../output.docx . -x ".*"
```

## Dependencies

- **pandoc**: Text extraction (`bash` to install/run)
- **docx**: `npm install -g docx` (new documents)
- **LibreOffice**: PDF conversion
- **Poppler**: `pdftoppm` for images

Use `bash` to install any missing dependencies and run all commands.

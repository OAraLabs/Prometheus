// make_iconset.swift: render the app icon's iconset from one square mark.
//
//   xcrun swift make_iconset.swift <mark.png> <out.iconset> <name:pixels> [<name:pixels> ...]
//
// The names and sizes come from build_app.py (ICONSET), so there is one list. Each file is the mark drawn
// into macOS's icon grid: a rounded square (corner radius 22.5% of the art) filling 824/1024 of the canvas,
// transparent around it, which is how Big Sur and later icons are shaped. macOS does not round an icon for
// you, and a bare square tile looks foreign next to every other app in Login Items and in notifications.
//
// Never upscales on purpose: build_app.py asks only for sizes the source can supply.

import CoreGraphics
import Foundation
import ImageIO

let arguments = CommandLine.arguments
guard arguments.count >= 4 else {
    FileHandle.standardError.write(Data("usage: make_iconset mark.png out.iconset name:pixels...\n".utf8))
    exit(2)
}

let sourceURL = URL(fileURLWithPath: arguments[1])
let outputDirectory = URL(fileURLWithPath: arguments[2], isDirectory: true)
guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
      let mark = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    FileHandle.standardError.write(Data("cannot read \(sourceURL.path)\n".utf8))
    exit(1)
}
try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

for spec in arguments.dropFirst(3) {
    let parts = spec.split(separator: ":")
    guard parts.count == 2, let pixels = Int(parts[1]), pixels > 0 else {
        FileHandle.standardError.write(Data("bad spec \(spec)\n".utf8))
        exit(2)
    }
    guard let context = CGContext(
        data: nil, width: pixels, height: pixels, bitsPerComponent: 8, bytesPerRow: 0,
        space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { exit(1) }
    context.interpolationQuality = .high

    let canvas = CGFloat(pixels)
    let art = canvas * 824.0 / 1024.0
    let inset = (canvas - art) / 2
    let rect = CGRect(x: inset, y: inset, width: art, height: art)
    let radius = art * 0.225
    context.addPath(CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil))
    context.clip()
    context.draw(mark, in: rect)

    guard let image = context.makeImage(),
          let destination = CGImageDestinationCreateWithURL(
              outputDirectory.appendingPathComponent(String(parts[0])) as CFURL, "public.png" as CFString, 1, nil)
    else { exit(1) }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { exit(1) }
}

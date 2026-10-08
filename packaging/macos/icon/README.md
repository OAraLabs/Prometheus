`mark-512.png` is the OAra "O." mark (512 x 512), the same file oara.ai serves as its icon. It is the
app icon by Will's decision (2026-10-08). `make_iconset.swift` draws it into macOS's icon shape at each size
`build_app.py` lists in `ICONSET`; `iconutil` turns the set into `Prometheus.icns`. The source is 512 px, so no
1024 px slot is produced (that would be an upscale); a higher-resolution mark would add the `@2x` 512 entry.

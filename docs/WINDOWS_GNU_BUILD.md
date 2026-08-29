# Building on Windows without MSVC

The default Windows target (`x86_64-pc-windows-msvc`) needs Visual
Studio Build Tools, which needs administrator rights to install. If
that's not available — locked-down corporate machine, no admin, etc.
— the `x86_64-pc-windows-gnu` target builds without it, using a
portable MinGW-w64 toolchain instead. This is the path actually used
to build and test `geopq-cli`. It works, with one non-obvious
consequence once you get there.

## 1. Rust + a portable MinGW-w64

```powershell
winget install --id Rustlang.Rustup -e
winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e --scope user
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```

Add the WinLibs `mingw64\bin` directory to `PATH` (winget prints the
exact path; it's under
`%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs...`).
Neither of these installs needs admin rights.

Both binaries build from here: `cargo build --release --bins`.

## 2. Debug builds may be too large to run

A debug build of the GUI (and, since it shares the same crate, of
`geopq-cli` too) can exceed 2.5 GB — full DWARF debug info across a
large dependency tree (`wgpu`, `datafusion`, `egui`, …). Past a
certain size, MinGW's PE linker has been observed to produce an
executable Windows refuses to run at all (`ERROR_BAD_EXE_FORMAT` /
"not a valid Win32 application", even though the file's PE header
inspects as well-formed). `--release` (already `lto = "thin"` +
`strip = true` in this repo's profile) avoids it — a release build of
the full GUI is ~150 MB, not ~2.8 GB. If you need to debug something
that only reproduces in a debug build, this is worth knowing before
assuming the crash you're chasing is a logic bug rather than a linker
artifact.

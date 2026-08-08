# Building on Windows without MSVC (and with `gdal-import`)

The default Windows target (`x86_64-pc-windows-msvc`) needs Visual
Studio Build Tools, which needs administrator rights to install. If
that's not available — locked-down corporate machine, no admin, etc.
— the `x86_64-pc-windows-gnu` target builds without it, using a
portable MinGW-w64 toolchain instead. This is the path actually used
to build and test the `feature/gdb-import` and `feature/cli` branches.
It works, but has three non-obvious steps.

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

A default (`gdal-import`-free) build works from here:
`cargo build --release`.

## 2. `gdal-import`: GDAL needs a real import library, not just the DLL

`gdal-sys`'s build script has a Windows-GNU-specific path that scans
`$GDAL_HOME/bin` for a `gdal*.dll` and links against it directly —
but plain `-l<dllname>` doesn't resolve for MinGW's `ld` the way the
build script assumes; it fails with `cannot find -lgdalNNN.dll`. A
static-style import library works, but has to be built by hand and
placed under a specific, hardcoded filename:

```bash
# From a directory containing (a copy of) the target gdalNNN.dll:
gendef gdalNNN.dll                                  # -> gdalNNN.def
dlltool -m i386:x86-64 -d gdalNNN.def -D gdalNNN.dll -l lib/gdal_i.lib
cp lib/gdal_i.lib lib/libgdal_i.a   # rustc's own static-lib search wants THIS name
```

Both filenames are required: `gdal-sys`'s build script only recognizes
a static lib named exactly `gdal_i.lib` (Windows/MSVC convention,
hardcoded in `gdal-sys/build.rs`), but `rustc` itself — independent of
that build script — resolves `-lstatic=gdal_i` by searching for
`libgdal_i.a` (GNU convention). Same archive, two names, one for each
consumer.

Then point the build at it:

```powershell
$env:GDAL_HOME = "C:\path\to\a\directory\containing\bin\ and\lib\"
$env:GDAL_LIB_DIR = "C:\path\to\that\lib"
$env:GDAL_VERSION = "3.12.0"   # match gdal-sys/prebuilt-bindings; must match the DLL's real version
cargo build --release --features gdal-import
```

If your OSGeo4W (or other GDAL) install has multiple `gdalNNN.dll`
versions side by side in one `bin` directory, isolate the one you
want into its own folder first — the build script's directory scan
picks whichever the OS happens to return first, which is not
guaranteed to be the newest.

## 3. Debug builds may be too large to run

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

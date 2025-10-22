# Harmoniq music workspace

This directory is reserved for Harmoniq assets when packaging the app as a PyInstaller executable. Place any shipped stems, example tracks, or temporary render outputs here so the runtime can locate them next to the executable.

The build scripts include this folder when constructing the desktop bundle, and `harmoniq_pywebview.py` ensures it exists at launch.

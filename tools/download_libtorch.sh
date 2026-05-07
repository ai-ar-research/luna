#!/bin/bash
curdir=$PWD
mydir="${0%/*}"
version=$1
cd $mydir

# Detect system
# Get OS name, and arhitecture
os=$(uname -s)
arch=$(uname -m)

echo "Downloading PyTorch"

# Choose the correct URL based on system
if [[ "$os" == "Darwin" && "$arch" == "arm64" ]]; then
    # macOS ARM64 
    # Apple Silicon Macs (M1/M2/M3)
    wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-$version.zip -O libtorch-$version.zip -q --show-progress --progress=bar:force:noscroll
elif [[ "$os" == "Darwin" ]]; then
    # macOS Intel
    wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-$version.zip -O libtorch-$version.zip -q --show-progress --progress=bar:force:noscroll
else
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$version%2Bcpu.zip -O libtorch-$version.zip -q --show-progress --progress=bar:force:noscroll
fi

echo "Unzipping PyTorch"
unzip libtorch-$version.zip >> /dev/null
mv libtorch libtorch-$version

echo "Successfully installed PyTorch $version for $os $arch"


# Check multiple locations for OpenMP (Open Multi-Processing)
#       An API that allows for multiple CPU cores to be used simultaneously
# PyTorch uses OpenMP for parallel computation
#       For matrix and tensor operations       
# Without OpenMP was getting "dyld: Library not loaded: @rpath/libomp.dylib" error, when running the test file
#       dyld is macOS's system component which loads shared libraries, resolves symbol dependencies between libraries and manages library paths + versions
# To resolve I copied the location of where OpenMP was to where Torch expected it to be (ie the commands at the bottom of file)  

# Can this be done more efficiently / or is this needed at all (ie is there a better way to either avoid the error all together or resolve the error)

if [[ "$os" == "Darwin" ]]; then
    echo "Checking OpenMP dependency for macOS..."

    openmp_lib=""
    openmp_paths=(
        "$(brew --prefix libomp 2>/dev/null)/lib/libomp.dylib"
        "/opt/homebrew/lib/libomp.dylib"
        "/usr/local/lib/libomp.dylib"
        "$CONDA_PREFIX/lib/libomp.dylib"
        "/opt/anaconda3/lib/libomp.dylib"
    )

    for path in "${openmp_paths[@]}"; do
        if [[ -f "$path" ]]; then
            echo "OpenMP found at: $path"
            openmp_lib="$path"
            break
        fi
    done

    if [[ -z "$openmp_lib" ]]; then
        echo "OpenMP not found. Attempting to install via Homebrew..."
        if command -v brew &> /dev/null && brew install libomp; then
            openmp_lib="$(brew --prefix libomp)/lib/libomp.dylib"
        elif command -v conda &> /dev/null && conda install -c conda-forge llvm-openmp -y; then
            openmp_lib="$CONDA_PREFIX/lib/libomp.dylib"
        else
            echo "Could not install OpenMP. Install manually: brew install libomp"
        fi
    fi

    if [[ -f "$openmp_lib" ]]; then
        echo "Copying OpenMP library from $openmp_lib to PyTorch directory..."
        cp "$openmp_lib" libtorch-$version/lib/libomp.dylib
        echo "OpenMP library copied successfully"
    else
        echo "WARNING: OpenMP library not found; tests may crash at runtime."
    fi
fi

cd $curdir
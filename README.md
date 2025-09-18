# gen_eth

## Usage

Generate vanity Ethereum address with prefix 0x1a2b

```bash
python .\pyvanityeth.py --prefix 0x1a2b
```

To display a help message listing the available command line arguments, run:

```bash
python .\pyvanityeth.py -h
```

## Installation

```bash
python -m pip install pycuda
python -m pip install numpy
python -m pip install python-decouple
python -m pip install pycryptodome
python -m pip install ecdsa
```

## Configuration

Configuration is done through a file called `settings.ini`. Example:

```ini
[settings]
CL_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
CUDA_DLL_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64
```

- CL_PATH - C++ compiler path
- CUDA_DLL_PATH - CUDA Toolkit DLLs path
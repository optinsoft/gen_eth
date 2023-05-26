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

Set the C++ compiler path in `settings.ini` (you can copy `settings.0.ini` to `settings.ini`), example:

```ini
[settings]
CL_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.35.32215\bin\Hostx64\x64
```

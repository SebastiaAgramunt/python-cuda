## Install package

Create a new virtual environment

```bash
rm -rf .venv
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

and install the package, this compiles the `*.cu` code

```bash
.venv/bin/python -m pip install .
```

## Launch a script

With your new environment launch a script to test

```bash
.venv/bin/python scripts/script.py
```


## Test it!

After installing the environment test with

```bash
.venv/bin/python -m pip install pytest
.venv/bin/pytest -v .
```

## Build the wheel

Everything is encoded in the script `scripts/build.sh`. Run all the steps as

```bash
ASK=install_environment  ./scripts/build.sh
TASK=run_tests  ./scripts/build.sh
TASK=build_wheel  ./scripts/build.sh
TASK=test_install_wheel  ./scripts/build.sh
TASK=cleanup  ./scripts/build.sh
```

No need to explain what each step does, it's self explanatory from the tasks. The repaired wheel will show in the `wheelhouse` with tag `manylinux_2_34_x86_64`. This would work for versions of GLIBC 2.34 and larger in `x86_64` machines.
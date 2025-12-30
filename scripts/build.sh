#!/bin/bash


set -vex

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname "$THIS_DIR")

ENV_NAME=.venv
ENV_NAME_TEST=.venv_test

install_environment(){
    rm -rf ${ROOT_DIR}/${ENV_NAME}
    python -m venv ${ROOT_DIR}/${ENV_NAME}
    ${ROOT_DIR}/${ENV_NAME}/bin/python -m pip install --upgrade pip
    ${ROOT_DIR}/${ENV_NAME}/bin/pip install ${ROOT_DIR}
}

# run tests with the environment created in install_environment
run_tests(){
    ${ROOT_DIR}/${ENV_NAME}/bin/python -m pip install pytest
    ${ROOT_DIR}/${ENV_NAME}/bin/python -m pytest ${ROOT_DIR}
}

# Build wheel and repair
build_wheel(){
    
    rm -rf ${ROOT_DIR}/dist
    rm -rf ${ROOT_DIR}/build

    # create blank environment
    rm -rf ${ROOT_DIR}/${ENV_NAME}
    python -m venv ${ROOT_DIR}/${ENV_NAME}
    ${ROOT_DIR}/${ENV_NAME}/bin/python -m pip install --upgrade pip

    # activate, install pkgs and build wheel
    source ${ROOT_DIR}/${ENV_NAME}/bin/activate
    pip install wheel pybind11 auditwheel repairwheel patchelf build
    pip install setuptools==70.3.0
    python -m build

    if [ $(arch) = "x86_64" ]; then
        platform="manylinux_2_34_x86_64"
    elif [ $(arch) = "aarch64" ]; then
        platform="manylinux_2_34_aarch64"
    else
        echo "ERROR: Unknown architecture"
        exit 1;
    fi

    auditwheel repair --exclude libcu* \
                      $(ls dist/*.whl | head -n 1) \
                      --plat ${platform} \
                      -w wheelhouse
}


test_install_wheel(){
    rm -rf ${ROOT_DIR}/${ENV_NAME_TEST}
    python -m venv ${ROOT_DIR}/${ENV_NAME_TEST}
    ${ROOT_DIR}/${ENV_NAME_TEST}/bin/pip install --upgrade pip
    ${ROOT_DIR}/${ENV_NAME_TEST}/bin/pip install --force-reinstall $(ls ${ROOT_DIR}/wheelhouse/*$(arch).whl | head -n 1)
    ${ROOT_DIR}/${ENV_NAME_TEST}/bin/python -c "import matmul"
}

cleanup(){
    rm -rf ${ROOT_DIR}/dist
    rm -rf ${ROOT_DIR}/build
    rm -rf ${ROOT_DIR}/tests/build
    rm -rf ${ROOT_DIR}/${ENV_NAME_TEST}
    rm -rf ${ROOT_DIR}/${ENV_NAME}
}

croak(){
    echo "[ERROR] $*" > /dev/stderr
    exit 1
}

main(){

  if [[ -z "$TASK" ]]; then
    croak "No TASK specified."
  fi
  echo "[INFO] running $TASK $*"
  $TASK "$@"
}

main "$@"

# TASK=install_environment  ./scripts/build.sh
# TASK=run_tests  ./scripts/build.sh
# TASK=build_wheel  ./scripts/build.sh
# TASK=test_install_wheel  ./scripts/build.sh
# TASK=cleanup  ./scripts/build.sh
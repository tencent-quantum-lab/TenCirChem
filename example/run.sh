#
# Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
# This file is distributed under ACADEMIC PUBLIC LICENSE
# and WITHOUT ANY WARRANTY. See the LICENSE file for details.
#

export PYTHONPATH=../:PYTHONPATH
code=0
for python_args in *.py; do
    # run water_pes.py requires CuPy
    if [ "$python_args" = "water_pes.py" ]; then
      continue
    fi
    if [ "$python_args" = "hea_geom_opt.py" ]; then
      pip install pyberny
    fi
    echo ============================$python_args=============================
    timeout 20s python $python_args
    exit_code=$?
    echo ============================$python_args=============================
    # if not the time out exit code or normal exit code
    if [ $exit_code -ne 124 ] && [ $exit_code -ne 0 ]; then
        echo "The script failed with exit code $exit_code" >&2
        code=1
    fi
done

exit $code
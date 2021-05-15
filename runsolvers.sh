# Enter the main project directory and use `bash runsolvers.sh` to run both solvers simultaneously

source ./venv/bin/activate

echo "With Splitting"
cd refactored_splitting
python3 initialisation.py
printf "\n \n"

echo "Without Splitting"
cd ../no-splitting
python3 example.py
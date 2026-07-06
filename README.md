[autoSAR_instructions_next_steps_with_ocean_contrast.md](https://github.com/user-attachments/files/29713713/autoSAR_instructions_next_steps_with_ocean_contrast.md)
# AutoSAR Setup Instructions

**Date:** 6/23/26  
**Purpose:** Set up Git, Homebrew, ESA SNAP/`esa_snappy`, the AutoSAR preprocessing repository, and the `auto_SAR_Ocean_Contrast` repository. This guide also includes PyCharm setup, Ocean Contrast package requirements, and troubleshooting notes from the problems encountered during setup.

---

## 1. Getting Git Activated

### 1.1 Install Git

Open Terminal and check whether Git is installed:

```bash
git --version
```

If prompted to install the command line developer tools, install them.

### 1.2 Create a GitHub account

Create an account at GitHub if you do not already have one.

### 1.3 Configure Git in Terminal

Run the following commands, replacing the name and email with your own GitHub information:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global --list
```

Confirm that the name and email you entered are listed.

Example:

```bash
git config --global user.name "erinreilly1234"
git config --global user.email "reillymerin@gmail.com"
git config --global --list
```

Expected output should include something like:

```text
user.name=erinreilly1234
user.email=reillymerin@gmail.com
```

---

## 2. Install Homebrew and GitHub CLI

### 2.1 Install Homebrew

Homebrew is useful for installing software from Terminal.

Run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, Homebrew may print **Next steps**. Follow those steps to add Homebrew to your PATH. On Apple Silicon Macs, the commands usually look like this:

```bash
echo >> /Users/ereilly/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/ereilly/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

If your username is different, replace `ereilly` with your own username.

### 2.2 Install GitHub CLI

Run:

```bash
brew install gh
```

Check the installation:

```bash
gh --version
```

You should see a GitHub CLI version number.

### 2.3 Log in to GitHub from Terminal

Run:

```bash
gh auth login
```

Respond like this:

```text
Where do you use GitHub? GitHub.com
What is your preferred protocol for Git operations on this host? HTTPS
Authenticate Git with your GitHub credentials? Yes
How would you like to authenticate GitHub CLI? Login with a web browser
```

Then log in and authorize GitHub in the browser.

---

## 3. Download Data from ASF Vertex

1. Go to <https://search.asf.alaska.edu/>.
2. Log in with your NASA Earthdata account.
3. Download the SAR images you need.
4. Save them in a folder on your computer.

Example input folder:

```text
/Users/ereilly/Documents/ASF_downloads
```

Keep paths simple and avoid special characters in folder names.

---

## 4. Set Up the `esa_snappy` Virtual Environment

Use this environment for SNAP/preprocessing work.

### 4.1 Confirm Apple Silicon architecture

Run:

```bash
uname -m
```

Expected output:

```text
arm64
```

### 4.2 Install Python 3.12

Run:

```bash
brew --version
brew update
/opt/homebrew/bin/brew install python@3.12
```

### 4.3 Create the `esa_snap_312` environment

Run:

```bash
/opt/homebrew/bin/python3.12 -m venv ~/venvs/esa_snap_312
source ~/venvs/esa_snap_312/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install esa_snappy
```

### 4.4 Configure SNAP Python interface

Run:

```bash
/Applications/esa-snap/bin/snappy-conf "$VIRTUAL_ENV/bin/python"
```

Expected output should look similar to:

```text
Configuring ESA SNAP-Python interface...
Found esa_snappy installed in '/Users/ereilly/venvs/esa_snap_312/lib/python3.12/site-packages'
Starting configuration...
Configuration finished successful!
Done. The SNAP-Python interface is located in '/Users/ereilly/venvs/esa_snap_312/lib/python3.12/site-packages/esa_snappy'
```

As long as it says **successful** and not **failed**, warnings can usually be ignored.

---

## 5. Pull the AutoSAR Code from GitHub

### 5.1 Clone the preprocessing repository

Go to the GitHub repository:

```text
https://github.com/erinreilly1234/autoSAR_preprocessing
```

Click the green **Code** button and copy the HTTPS URL.

In Terminal, move to the directory where you want the repository saved:

```bash
cd ~/Documents
```

Clone the repository:

```bash
git clone https://github.com/erinreilly1234/autoSAR_preprocessing.git
```

### 5.2 Clone the Ocean Contrast repository

Repeat the same process for:

```text
https://github.com/erinreilly1234/auto_SAR_Ocean_Contrast
```

Run:

```bash
cd ~/Documents
git clone https://github.com/erinreilly1234/auto_SAR_Ocean_Contrast.git
```

---

## 6. Open the AutoSAR Code in PyCharm

Start here after Section 5 is finished and both repositories are cloned onto your computer.

For preprocessing, keep using the virtual environment created in Section 4. The Ocean Contrast repository needs a separate `RS` environment because it uses GDAL and the `osgeo` Python bindings.

### 6.1 Install and open PyCharm

1. Install PyCharm Community or Professional.
2. Open PyCharm.
3. Choose **Open**.
4. Select the cloned preprocessing repository folder:

```text
/Users/ereilly/Documents/autoSAR_preprocessing
```

If PyCharm asks whether you trust the project, click **Trust Project**.

### 6.2 Connect PyCharm to the `esa_snap_312` environment

In PyCharm, go to:

```text
PyCharm > Settings or Preferences > Project: autoSAR_preprocessing > Python Interpreter
```

Then choose:

```text
Add Interpreter > Add Local Interpreter > Existing environment
```

Select this Python interpreter:

```text
/Users/ereilly/venvs/esa_snap_312/bin/python
```

Click **OK** or **Apply**.

If the file chooser hides folders, press:

```text
Command + Shift + .
```

### 6.3 Test the environment inside PyCharm

Open the PyCharm Terminal and run:

```bash
which python
python --version
python -c "import esa_snappy; print('esa_snappy import successful')"
```

Expected results:

- `which python` should include `venvs/esa_snap_312`.
- `python --version` should show Python 3.12.
- The import test should print `esa_snappy import successful`.

---

## 7. Set Up Input Data and Output Folders

### 7.1 Confirm where ASF Vertex downloads are saved

Use the folder where you saved the SAR images from ASF Vertex.

Example input folder:

```text
/Users/ereilly/Documents/ASF_downloads
```

### 7.2 Create an output folder

In the PyCharm Terminal, run:

```bash
mkdir -p ~/Documents/autoSAR_outputs
```

### 7.3 Use full paths in code or run configurations

Example paths:

```python
input_folder = "/Users/ereilly/Documents/ASF_downloads"
output_folder = "/Users/ereilly/Documents/autoSAR_outputs"
```

If the repository has a README or config file, check it before running so your input and output paths match what the script expects.

---

## 8. Run the Preprocessing Code in PyCharm

### 8.1 Find the Python file you want to run

In the Project panel on the left, open the `autoSAR_preprocessing` folder and look for:

- the main script,
- a `scripts` folder,
- a README file,
- or instructions from the repository.

### 8.2 Run the file

Right-click inside the Python file and choose **Run**. You can also use the green Run button at the top of PyCharm.

### 8.3 Add input arguments, if needed

If the script needs command-line input arguments, make a Run Configuration:

```text
Run > Edit Configurations...
```

Set the configuration like this:

```text
Script path: choose the .py file you want to run
Python interpreter: /Users/ereilly/venvs/esa_snap_312/bin/python
Working directory: /Users/ereilly/Documents/autoSAR_preprocessing
Parameters: add any required input/output arguments
```

Example parameters, only if the script uses command-line arguments:

```text
--input "/Users/ereilly/Documents/ASF_downloads" --output "/Users/ereilly/Documents/autoSAR_outputs"
```

If the script does not use command-line arguments, leave Parameters blank and set paths directly in the code or config file.

### 8.4 Watch the Run window

The Run window at the bottom of PyCharm will show progress messages, errors, and completion messages. When the run finishes, check the output folder for new files.

---

## 9. Set Up a Separate `RS` Environment for `auto_SAR_Ocean_Contrast`

Start this section after both repositories have been cloned.

Use this `RS` environment for the Ocean Contrast script. Keep using `esa_snap_312` for SNAP/preprocessing work.

Important: Do **not** use the Conda `base` environment for this part. If your prompt starts with `(base)`, run:

```bash
conda deactivate
```

Why a separate environment is needed:

- The preprocessing environment needs `esa_snappy`.
- The Ocean Contrast code needs GDAL/`osgeo` and the local `OilClassification` package.
- Keeping them separate avoids package conflicts.

Python version note: the repository install notes say the code was tested on an M1 Mac with Python 3.9 and may work with later versions. The commands below use Python 3.11 because the troubleshooting log showed Python 3.11/GDAL on this machine. The main rule is to avoid mixing Python versions inside the same virtual environment.

### 9.1 Create a clean `RS` environment

Open Terminal and run:

```bash
cd ~/Documents

# Only remove RS if it is broken or was created with the wrong Python version.
rm -rf RS

/opt/homebrew/bin/python3.11 -m venv RS
source RS/bin/activate
python --version
which python
python -m pip --version
```

If Python 3.11 is not installed, install it first:

```bash
brew install python@3.11
```

Expected result:

- Your prompt should start with `(RS)`.
- `which python` should point to `/Users/ereilly/Documents/RS/bin/python`.

### 9.2 Install the Ocean Contrast Python packages

These packages are needed by the `auto_SAR_Ocean_Contrast` `setup.py` file and by the Ocean Contrast script imports.

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy matplotlib netCDF4 PyYAML spectral scipy scikit-image jupyter jupyterlab
```

Required Ocean Contrast package list:

```text
numpy
matplotlib
netCDF4
PyYAML
spectral
scipy
scikit-image
jupyter
jupyterlab
GDAL
```

### 9.3 Install GDAL correctly

GDAL is the package that provides this import:

```python
from osgeo import gdal
```

Do **not** install a package named `osgeo`.

Run these commands as separate lines:

```bash
brew install gdal
GDAL_VERSION=$(gdal-config --version)
echo $GDAL_VERSION
python -m pip install --no-cache-dir --force-reinstall "GDAL==$GDAL_VERSION"
```

Important: Do not paste `brew install`, `GDAL_VERSION`, and `pip install` onto one command line.

### 9.4 Install the local `OilClassification` package

This lets Python import `OilClassification` from anywhere, including PyCharm run configurations.

```bash
cd ~/Documents/auto_SAR_Ocean_Contrast
python -m pip install -e . --no-deps
```

### 9.5 Test the `RS` environment

Run this test from Terminal while the `RS` environment is active:

```bash
python -c "import numpy, matplotlib, netCDF4, yaml, spectral, scipy, skimage; from osgeo import gdal; import OilClassification.io as io; print('Ocean Contrast environment works'); print(gdal.VersionInfo())"
```

If the command prints `Ocean Contrast environment works` and a GDAL version number, the packages are installed in the correct environment.

---

## 10. Run the Ocean Contrast Script in PyCharm

### 10.1 Open the Ocean Contrast repository

In PyCharm, choose:

```text
File > Open
```

Select:

```text
/Users/ereilly/Documents/auto_SAR_Ocean_Contrast
```

If PyCharm asks whether to trust the project, click **Trust Project**.

### 10.2 Select the `RS` interpreter

Go to:

```text
PyCharm > Settings or Preferences > Project: auto_SAR_Ocean_Contrast > Python Interpreter
```

Then choose:

```text
Add Interpreter > Add Local Interpreter > Existing environment
```

Select this interpreter:

```text
/Users/ereilly/Documents/RS/bin/python
```

In the PyCharm Terminal, confirm the interpreter:

```bash
source ~/Documents/RS/bin/activate
which python
python --version
python -c "from osgeo import gdal; import OilClassification.io as io; print('works')"
```

### 10.3 Prepare the case folder and `config.yaml`

The Ocean Contrast script expects a case directory. That case directory must contain a `config.yaml` file. The repository has prototype `config.yaml` files that can be copied, edited, and renamed to `config.yaml`.

Example layout:

```text
/Users/ereilly/Documents/ocean_contrast_cases/my_case/
    config.yaml
    input_files/
    output/
```

Use full paths in `config.yaml` where possible. Check capitalization and folder names carefully.

### 10.4 Make a PyCharm Run Configuration

Go to:

```text
Run > Edit Configurations...
```

Set the run configuration like this:

```text
Script path:
/Users/ereilly/Documents/auto_SAR_Ocean_Contrast/auto_calc_contrast_in_ocean_all_formats.py

Python interpreter:
/Users/ereilly/Documents/RS/bin/python

Working directory:
/Users/ereilly/Documents/auto_SAR_Ocean_Contrast

Parameters:
/Users/ereilly/Documents/ocean_contrast_cases/my_case
```

Script choice: If the repository instructions or your advisor specify `auto_calc_contrast_in_ocean.py` instead of `auto_calc_contrast_in_ocean_all_formats.py`, use that file as the Script path. The setup and package requirements are the same.

Debug options:

```text
-d    Print more details and save intermediate figures.
-I    Display figures interactively.
```

For example:

```text
/Users/ereilly/Documents/ocean_contrast_cases/my_case -d
```

### 10.5 Run and check outputs

Click the green Run button. Watch the Run window at the bottom of PyCharm. When the run finishes, check the output directory listed in `config.yaml`.

---

## 11. Update Code and Fetch Git Tags Before Running

The Ocean Contrast code builds a version label using:

```bash
git describe --abbrev=0 --tags
```

If your clone has no tags, the script can fail with:

```text
fatal: No names found, cannot describe anything.
```

Before a new run, update both repositories:

```bash
cd ~/Documents/autoSAR_preprocessing
git pull

cd ~/Documents/auto_SAR_Ocean_Contrast
git pull
```

Then fetch tags for `auto_SAR_Ocean_Contrast`:

```bash
cd ~/Documents/auto_SAR_Ocean_Contrast
git remote -v
git fetch origin --tags
git fetch https://github.com/nasa-jpl/auto_SAR_Ocean_Contrast.git --tags
git tag
git describe --abbrev=0 --tags
```

If `git describe` prints a tag number, the version-label problem is fixed.

Also make sure the PyCharm Run Configuration working directory is the repository root:

```text
/Users/ereilly/Documents/auto_SAR_Ocean_Contrast
```

---

## 12. Problems Encountered and How to Fix Them

### Problem 1: Homebrew says `No available formula with the name gdal_version=3.13.1`

This happened because multiple commands were pasted onto one line:

```bash
brew install gdal GDAL_VERSION=$(gdal-config --version) pip install "GDAL=$GDAL_VERSION"
```

Fix: run the commands separately and use `==` for the pip package version:

```bash
brew install gdal
GDAL_VERSION=$(gdal-config --version)
echo $GDAL_VERSION
python -m pip install "GDAL==$GDAL_VERSION"
```

### Problem 2: `pip install osgeo` fails

Do not install `osgeo` directly. The import name is `osgeo`, but the package name to install is `GDAL`.

Fix:

```bash
python -m pip uninstall -y osgeo
GDAL_VERSION=$(gdal-config --version)
python -m pip install --force-reinstall "GDAL==$GDAL_VERSION"
```

Then test:

```bash
python -c "from osgeo import gdal; print(gdal.VersionInfo())"
```

### Problem 3: `ModuleNotFoundError: No module named osgeo` even after installing GDAL

This usually means GDAL was installed into a different Python environment than the one running the code. In the troubleshooting log, `python -m pip` and `pip` pointed to different Python versions at one point.

Check the environment:

```bash
source ~/Documents/RS/bin/activate
which python
which pip
python --version
python -m pip --version
pip --version
```

Fix: always install packages with `python -m pip`, not plain `pip`.

If the environment is mixed or confusing, delete and recreate `RS`:

```bash
deactivate
rm -rf ~/Documents/RS
/opt/homebrew/bin/python3.11 -m venv ~/Documents/RS
source ~/Documents/RS/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Problem 4: The prompt shows `(base)` instead of `(RS)`

This means Conda is active. The Ocean Contrast setup in this guide should be run from the `RS` virtual environment, not Conda `base`.

Fix:

```bash
conda deactivate
source ~/Documents/RS/bin/activate
```

### Problem 5: `fatal: No names found, cannot describe anything`

This error comes from:

```bash
git describe --abbrev=0 --tags
```

It means the current clone does not have tags available, or PyCharm is running the script from the wrong working directory.

Fix the tags first:

```bash
cd ~/Documents/auto_SAR_Ocean_Contrast
git fetch origin --tags
git fetch https://github.com/nasa-jpl/auto_SAR_Ocean_Contrast.git --tags
git tag
git describe --abbrev=0 --tags
```

Then fix PyCharm:

```text
Run > Edit Configurations...
Working directory: /Users/ereilly/Documents/auto_SAR_Ocean_Contrast
```

### Problem 6: `ModuleNotFoundError: No module named OilClassification`

Python cannot see the local package from the repository.

Fix:

```bash
source ~/Documents/RS/bin/activate
cd ~/Documents/auto_SAR_Ocean_Contrast
python -m pip install -e . --no-deps
python -c "import OilClassification.io as io; print('OilClassification works')"
```

### Problem 7: The script cannot find `config.yaml` or input files

Make sure Parameters points to the case directory, not the repository directory, and make sure that case directory contains `config.yaml`.

Parameters example:

```text
/Users/ereilly/Documents/ocean_contrast_cases/my_case
```

Make sure the PyCharm Working directory is still the repository root:

```text
/Users/ereilly/Documents/auto_SAR_Ocean_Contrast
```

---

## 13. Quick Checklist Before Running

### For preprocessing/SNAP

- PyCharm interpreter is `/Users/ereilly/venvs/esa_snap_312/bin/python`.
- `esa_snappy` imports successfully.
- Input folder points to the ASF downloads.
- Output folder exists.

### For Ocean Contrast

- PyCharm interpreter is `/Users/ereilly/Documents/RS/bin/python`.
- Do not run Ocean Contrast from Conda `base`.
- This command works in the `RS` environment:

```bash
python -c "from osgeo import gdal; print(gdal.VersionInfo())"
```

- This command works in the `RS` environment:

```bash
python -c "import OilClassification.io as io; print('OilClassification works')"
```

- GDAL was installed with:

```bash
python -m pip install "GDAL==$GDAL_VERSION"
```

- `git describe --abbrev=0 --tags` works from:

```text
~/Documents/auto_SAR_Ocean_Contrast
```

- PyCharm Working directory is:

```text
/Users/ereilly/Documents/auto_SAR_Ocean_Contrast
```

- Parameters points to a case folder that contains `config.yaml`.
- Input and output paths in `config.yaml` are full paths and are spelled exactly correctly.

---

## 14. Clean Command Sheet for the `RS` Environment

Use this block for a clean setup after cloning `auto_SAR_Ocean_Contrast`:

```bash
cd ~/Documents
rm -rf RS
/opt/homebrew/bin/python3.11 -m venv RS
source RS/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy matplotlib netCDF4 PyYAML spectral scipy scikit-image jupyter jupyterlab
brew install gdal
GDAL_VERSION=$(gdal-config --version)
python -m pip install --no-cache-dir --force-reinstall "GDAL==$GDAL_VERSION"
cd ~/Documents/auto_SAR_Ocean_Contrast
python -m pip install -e . --no-deps
git fetch https://github.com/nasa-jpl/auto_SAR_Ocean_Contrast.git --tags
python -c "from osgeo import gdal; import OilClassification.io as io; print('success', gdal.VersionInfo())"
```

---

## 15. Source Notes

- The package list comes from `auto_SAR_Ocean_Contrast/setup.py`, which lists `numpy`, `matplotlib`, `PyYAML`, `spectral`, `gdal`, `netCDF4`, `jupyter`, `jupyterlab`, `scipy`, and `scikit-image` as install requirements.
- The repository install notes say the GeoTIFF interface requires `libgdal` 3.8.4 or later and that the code was tested on an M1 Mac with Python 3.9.
- The repository running instructions say the stand-alone Python script is easier for processing multiple files and uses a case directory containing `config.yaml`.
- The troubleshooting fixes above reflect the errors captured in the pasted Terminal log: the GDAL one-line command error, the failed `pip install osgeo` attempt, the Python/pip mismatch, and the missing Git tags error.

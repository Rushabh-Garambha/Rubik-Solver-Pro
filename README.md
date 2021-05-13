# Rubik Solver Pro

This project aims to help user solve rubik's cube by showing moves in form of arrow on the cube.

## Installation

Clone this repo in your system. And, install required libraries from PyPI:

```bash
pip install opencv-python
pip install numpy
pip install rubik-solver
```
## Run
execute `main.py` to run the program.
## Usage
There are two main steps:
1) Detecting Complete Cube
2) Solving Cube

### Detecting Complete Cube
During cube detection, main thing to keep is mind is cube orientation.
* While detecting side with `Yellow Center`, side with `Red Center` should be `Bottom` facing
* While detecting side with `White Center`, side with `Red Center` should be `Top` facing
* While detecting any other sides, side with `Yellow Center` should be `Top` facing and `White Center` should be `Bottom Facing`

When the cube face is detected, face preview will be visible in the preview slot. User can append detected side to cube 
state by pressing `a`. <br>
### Solving Cube
After complete cube is detected user can press `s` to solve the cube and retrieve the solving steps.
To solve the cube, first user have to start with showing the side with `Red Center` while side with `Yellow Center` faces `Top` and 
side with `White Center` faces `Bottom`.


# Rubik Solver Pro

This project aims to help user solve rubik's cube by showing moves in form of arrow on the cube.

## Installation

Clone this repo in your system and install required libraries from PyPI:

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

### Preview Images
#### Detecting Cube State
![CV SS 1](https://user-images.githubusercontent.com/89860786/206485326-ed29937f-c1f4-431b-96fe-a7c5896b97bc.png)
### Display Solution
![CV SS2](https://user-images.githubusercontent.com/89860786/206485316-6e656978-7479-43d7-b134-3215094f550c.png) ![CV SS3](https://user-images.githubusercontent.com/89860786/206485303-d64b1111-ab9d-470d-8faf-607c3b17ddfd.png) ![CV SS4](https://user-images.githubusercontent.com/89860786/206485262-d313df12-2370-4bac-b350-616b8061ad15.png)
### Cube Solved!
![image](https://user-images.githubusercontent.com/89860786/206485144-6c79cce7-304c-4709-9956-3040d3b98f43.png)


## FishML

### IMPORTANT
Our program might take a while to load during the very first time you run the program. When you first run, the model hasn't been trained yet. So it will take a while for the model to train. Subsequently after you run the program for the first time, it should take faster to run as the model has already been trained.

### Enhancing Fish Disease Diagnosis through Machine Learning

Through this solution, we intend to benefit 2 main entities: fish owners and lastly the fishes themselves.

We want to be able to speed up the processes of identifying diseases in fish such that it is able to seek treatment much faster and relieve the stress for fish owners to constantly take care after their fishes. 

## Python Packages Used
- `os` (standard library)
- `numpy`
- `streamlit`
- `tensorflow`
- `PIL` (Python Imaging Library, installed via `Pillow`)

## Installation
`pip install numpy streamlit tensorflow Pillow` -> This installs ALL the packages required for our program.

### NOTE
In our program there's a specific segment which reuqires customistation on your end. As the path to our datasets will be different from yours, it will require some changing.<br />
Whenever the code requires for you to specify a path, for example:<br />`base_dir = '/Users/aathithyaj/Desktop/GitHub/fish-disease-diagnosis-py/Dataset.csv'`<br />You should replace it with:<br />`base_dir = '/Users/{yourusername}/path/to/fish-disease-diagnosis-py/Dataset.csv'`<br />^ Change it to the path in which our program is found on your computer.

## Step by Step Guide:
### Step 1:
Open our program (with datasets and main.py file) in your desired code editor.
![image](https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/6b2d9bda-56c4-4299-98b3-bb7ec4cd5576)

### Step 2:
Open Main.py
![image](https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/33453b80-0367-4896-ada4-5d03af46c310)

### Step 3:
In your terminal go to the directory and type `streamlit run main.py`. Click enter.
![image](https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/6427b73e-4831-4196-96d2-52531e379b58)

### Step 4:
You should see a screen pop up. If you don't, manually click on the links displayed in the terminal after completing Step 3.
![image](https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/edf88c3a-32b7-4a6b-a5f3-add77b76f61e)

### Step 5:
After the program runs, you should see a `model_trained.h5` file which is basically our trained model.
![image](https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/ee382dcd-4b61-454c-9cd7-2cba3b97ac74)







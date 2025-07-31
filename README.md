ML Ops Major Exam\Assignment


Submitted To:
-------------
Dr Pratik Mazumder: pratikm@iitj.ac.in
Shubham Bagwari: p22cs201@iitj.ac.in
Divyaansh Mertia: m23cse013@iitj.ac.in

Submitted By:
-------------
Nitin Awasthi: g24ai1009@iitj.ac.in


Objective:
---------
Build a complete MLOps pipeline focused on Linear Regression only, integrating training, testing,
quantization, Dockerization, and CI/CD — all managed within a single main branch.

1. Repository Setup (As Per Submission Guidelines):
   -----------------------------------------------

  1.1 Setup Project Environment

       mkdir mlops-major-exam
       cd mlops-major-exam
       mkdir src tests .github .github/workflows

  1.2 Created Conda Environment

      conda create -n mlops3 python=3.9 -y
      conda activate mlops3

  1.3 Installed Dependencies

      pip install scikit-learn joblib numpy pytest
      pip freeze > requirements.txt

2. GitHub Setup
   ------------

  2.1 Initialize Git & Connect to Remote Repo
  
       git init
       git remote add origin https://github.com/rflinkbudget/mlops-major-exam.git

  2.2 Add .gitignore
       
       # Byte-compiled / cache
       __pycache__/
       *.py[cod]
       *.pyo

       # Model files
       *.joblib
       *.pth

3. Main Branch Folder Structure:
   mlops-major-exam/

├── src/

│ ├── train.py

│ ├── quantize.py

│ ├── predict.py

│ └── utils.py

├── tests/

│ └── test_train.py

├── Dockerfile

├── requirements.txt

└── .github/workflows/ci.yml
   

4. Project Initialization
-------------------------

  3.1 Organized code into `src/` and `tests/`
  3.2 Initialized Git and GitHub repo with CI/CD

5. Model Training (`train.py`)
------------------------------

  5.1 Loaded California Housing data
  5.2 Trained `LinearRegression` model
  5.3 Evaluated using R² and MSE
  5.4 Saved model using `joblib`

6. Unit Testing (`test_train.py`)
---------------------------------

  6.1 Verified data loading
  6.2 Checked model training and saving
  6.3 Ensured R² is above a minimum threshold


7. Manual Quantization (`quantize.py`)
--------------------------------------

   7.1 Quantized model weights with:

       `uint8`: incorrect due to negative values
       `int16`: accurate and preserves precision

   7.2 Compared original vs quantized predictions (Quantization Comparison Table)

      | Sample | Original Prediction | Quantized (`uint8`)  | Quantized (`int16`)   |
      |--------|----------------------|---------------------|---------------------- |
      | 0      | 0.7191               | -1943.47            | 0.7190                |
      | 1      | 1.7640               | -1962.79            | 1.7639                |
      | 2      | 2.7097               | -1911.62            | 2.7096                |
      | 3      | 2.8389               | -1975.52            | 2.8388                |
      | 4      | 2.6047               | -1964.39            | 2.6046                |

   7.3 Explanation

     i.  "uint8" only allows values between 0–255.
     ii.  Model coefficients include **negative values**, so casting to `uint8` caused **clipping/wrapping**.
     iii. This led to completely incorrect predictions.
     iv.  supports **signed values** and a larger range, preserving both sign and scale.
          Hence, `int16` was chosen for final quantization.

8. Dockerization

   Created `Dockerfile` to install deps, train, and run prediction
   Used `predict.py` for validation inside container

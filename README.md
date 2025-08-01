ML Ops Major Exam\Assignment
----------------------------

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

2. GitHub Setup:
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

3. Main Branch Folder Structure (mlops-major-exam):
   -----------------------------------------------
   
   ```text
   mlops-major-exam/
   ├── .github/
   │   └── workflows/
   │       └── ci.yml
   │
   ├── src/
   │   ├── train.py
   │   ├── quantize.py
   │   ├── predict.py
   │   └── utils.py
   │
   ├── tests/
   │   └── test_train.py
   │
   ├── Dockerfile
   ├── requirements.txt
   ├── .gitignore
   └── README.md



4. Project Initialization:
   ----------------------

      4.1 Organized code into `src/` and `tests/`
  
      4.2 Initialized Git and GitHub repo with CI/CD

5. Model Training (`train.py`):
   ---------------------------

      5.1 Loaded California Housing data
  
      5.2 Trained `LinearRegression` model
  
      5.3 Evaluated using R² and MSE
  
      5.4 Saved model using `joblib`

6. Unit Testing (`test_train.py`):
   ------------------------------

      6.1 Verified data loading
  
      6.2 Checked model training and saving
  
      6.3 Ensured R² is above a minimum threshold


7. Manual Quantization (`quantize.py`):
   -----------------------------------
   
	7.1 Model Size Comparison Table
		---------------------------
		
		This table compares the storage size of the original model vs. its quantized version.
        Quantization significantly reduces model size by converting weights to 8-bit integers.

		| Model                   | Size (KB) |
		| ----------------------- | --------- |
		| Original model (`.pth`) | 0.67 KB   |
		| Quantized model (8-bit) | 0.51 KB   |

	7.2 Quantization Accuracy
		---------------------
		
		This section evaluates how much error was introduced during quantization.
        Despite minimal parameter-level errors, the prediction shift remains within acceptable bounds.

		| Metric                     | Value        |
		| -------------------------- | ------------ |
		| Max Coefficient Error      | 0.000039     |
		| Intercept Error            | 0.000039     |
		| Max Prediction Difference  | 0.629886     |
		| Mean Prediction Difference | 0.054151     |
		| Quantization Quality       | Acceptable   |

	 7.3 Performance Metrics Comparison Table
		 ------------------------------------
		 
		Shows how the quantized model performs compared to the original trained model.
        Both R² and MSE are nearly the same, indicating the quantized model preserves accuracy well.
	 
		| Metric   | Train (Original) | Quantized |
		| -------- | ---------------- | --------- |
		| R² Score | 0.5758           | 0.5724    |
		| MSE      | 0.5559           | 0.5603    |


	7.4 Coefficient Comparison Table
		----------------------------
		
		Displays original and dequantized coefficients side by side.
        Minor differences confirm that 8-bit quantization preserved model parameters accurately.

		| Index         | Original Coef  | Dequantized Coef |
		| ------------- | -------------- | ---------------- |
		| 0             | 0.448675       | 0.448714         |
		| 1             | 0.009724       | 0.009763         |
		| 2             | -0.123323      | -0.123284        |
		| 3             | 0.783145       | 0.783184         |
		| 4             | -0.000002      | 0.000037         |
		| 5             | -0.003526      | -0.003487        |
		| 6             | -0.419792      | -0.419753        |
		| 7             | -0.433708      | -0.433669        |
		| **Intercept** | **-37.023278** | **-37.023239**   |

	7.5 Prediction Comparison (First 5 Samples) Table
		----------------------------------------------
		
		Compares predictions from the original and quantized models on sample data.
        Slight prediction differences validate that quantization doesn’t significantly impact output quality.

		| Index | Original Prediction | Quantized Prediction |
		| ----- | ------------------- | -------------------- |
		| 0     | 0.7191              | 0.7719               |
		| 1     | 1.7640              | 1.8237               |
		| 2     | 2.7097              | 2.7602               |
		| 3     | 2.8389              | 2.9038               |
		| 4     | 2.6047              | 2.6449               |

8. Dockerization:
   -------------

       Created `Dockerfile` to install deps, train, and run prediction
   
       Used `predict.py` for validation inside container
   
       #Build Docker Image
   
       docker build -t mlops-lr .
   
       #Running the container
   
       docker run --rm mlops-lr
    
       #Container Output
   
       2025-07-31 21:49:57 ✅ Sample Predictions: [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725]


9. CI/CD Workflow:
   --------------
       .github/workflows/ci.yml` defines 3 jobs:
   
       i.   test-suite(PyTest): This job runs all the unit tests written for model training. It checks if the data loads correctly, the model trains properly, and the saved model meets basic quality (like R² score).
                                It ensures your core logic is working before moving forward.

       ii.  train-and-quantize: Once tests pass, this job trains the model again and performs manual quantization using both uint8 and int16. 
                                It validates the quantization process and prints predictions to compare with the original model — a key requirement in the assignment.
						  
       iii. build-and-test-container: Finally, this job builds a Docker image and runs your predict.py inside the container. 
                                      It confirms that your entire pipeline — from training to inference — works in a portable, reproducible environment which is critical for MLOps deployment.



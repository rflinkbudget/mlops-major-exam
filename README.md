**ML Ops Major Exam\\Assignment**





**Submitted To:**

---

Dr Pratik Mazumder: pratikm@iitj.ac.in

Shubham Bagwari: p22cs201@iitj.ac.in

Divyaansh Mertia: m23cse013@iitj.ac.in



**Submitted By:**

---

Nitin Awasthi: g24ai1009@iitj.ac.in





**Objective:**

---

Build a complete MLOps pipeline focused on Linear Regression only, integrating training, testing,

quantization, Dockerization, and CI/CD — all managed within a single main branch.



**1. Repository Setup (As Per Submission Guidelines):**

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



**2. GitHub Setup**

**------------**





**mlops-major-exam/**

**├── src/**

**│ ├── train.py**

**│ ├── quantize.py**

**│ ├── predict.py**

**│ └── utils.py**

**├── tests/**

**│ └── test\_train.py**

**├── Dockerfile**

**├── requirements.txt**

**└── .github/workflows/ci.yml**






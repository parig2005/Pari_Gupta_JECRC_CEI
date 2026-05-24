📘 ML Foundations — Week 1 Assignment
a complete notebook covering the core math and programming fundamentals you need before getting into actual machine learning. nothing fancy, just making sure the basics are solid.

🗂️ what's covered
Part 1 — Python Fundamentals

control flow and data types (classify_number)
data structures — dicts, sets, list comprehensions
exception handling with safe_divide
higher order functions and lambdas (apply_twice)

Part 2 — NumPy

array creation, reshaping, indexing, slicing
element-wise vs matrix operations, dot products
SVD decomposition and rank-1 approximations

Part 3 — Pandas

Series vs DataFrame, .iloc vs .loc
filtering, groupby, aggregations
handling missing data (fill + drop strategies)

Part 4 — Linear Algebra

vector norms and matrix operations
eigenvalues and eigenvectors (with geometric intuition)
SVD → PCA connection

Part 5 — Statistics

descriptive stats: mean, median, std, IQR
hypothesis testing: one-sample t-test, Pearson correlation
regression metrics: MAE, MSE, RMSE, R², Adjusted R²
KS test, ADF stationarity test
PSI for distribution shift detection

Part 6 — Probability

basic probability, joint and conditional probability
Bayes' theorem and Naive Bayes from scratch
Normal, Binomial, Poisson distributions
Central Limit Theorem demo with sampling


🛠️ setup
bashpip install numpy pandas matplotlib scipy statsmodels
then just run the notebook top to bottom. all cells are dependent on each other so dont skip the setup cell.

📁 structure
week1/
│
├── week1_assignment.ipynb   # main notebook with all solutions
└── README.md                # this file

💡 key concepts (quick reference)
eigenvectors — special directions where a matrix only stretches/squishes a vector, no rotation. eigenvalue tells you by how much.
SVD — splits matrix X into U, S, Vt. the rows of Vt are your principal components (directions of max variance). this is literally what PCA does under the hood.
CLT — no matter how weird your population distribution looks, sample means will always form a normal distribution if n is large enough. this is why t-tests and z-tests work even on non-normal data.
PSI thresholds — below 0.1 is stable, 0.1–0.2 is minor drift worth watching, above 0.2 is a red flag and you should probably retrain your model.
concept drift vs covariate drift — concept drift is when the X→Y relationship changes. covariate drift is when the input distribution changes but the relationship stays the same.

✅ passing all assertions
every section has assertion blocks. if something fails check:

missing data is handled before any stats calculations (use df_filled not df)
SVD reconstruction uses full_matrices=True and slices U correctly
PSI adds epsilon before log to avoid log(0)


📌 notes
all random seeds are fixed (np.random.seed(...)) so results should be reproducible. if you're getting different numbers double check you ran the seed cell first.

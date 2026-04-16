Currently kencarpgersh5.py implements exactly the same solver as kencarp5.py. Convert it to a dynamic IMEX solver that takes a single ode_fn and at every step splits the function into implicit and explicit terms using Gershgorin's circle therem. Please write a docstring as part of the solver files that describes the method in detail, what it aims to do and in what situations it is better than the ordinary kencarp5.py method.

This will involve: replacing the upfront implicit and explicit terms with a single provided ode_fn, replacing _explicit_batched and _implicit_batched with combined _ode_batched and _implicit_jac_fn and _implicit_jac_batched with _jac_fn and _jac_batched. The batched jacobian should be evaluated at the start of every _step_batch, and the circle theorem should be applied to this Jacobian as follows:

The **Gershgorin Circle Theorem** is a powerful shortcut because it allows us to estimate the "danger zone" of a matrix's eigenvalues without actually solving a characteristic equation. When we use it to partition a Jacobian matrix $M$, we are essentially sorting the rows of the system into "fast" and "slow" categories.

Here is the step-by-step breakdown of how you would perform this partition.

## 1. The Core Logic: Estimating Local "Speed"

Recall that for each row $i$ of your Jacobian matrix $M$, the eigenvalues $\lambda$ must lie within a disk $D_i$ centered at the diagonal element $M_{ii}$ with a radius $R_i$ equal to the sum of the absolute values of the off-diagonal elements:

$$
R_i = \sum_{j \neq i} |M_{ij}|
$$

The "worst-case" eigenvalue associated with row $i$ is roughly $|M_{ii}| + R_i$. If this value is large, that specific row represents a variable (or a coupling) that evolves very quickly, potentially destabilizing an explicit solver.

## 2. The Partitioning Algorithm

### Step 1: Define your "Stability Speed Limit" (τ)

Before you look at the matrix, you must decide your threshold $\tau$. This is usually determined by your desired time step $\Delta t$ and the stability region of the explicit solver you *want* to use.

- For example, if you're using Forward Euler, you might set $\tau = 2/\Delta t$.

### Step 2: Calculate Row Magnitudes

For every row $i = 1, \dots, n$:

1. **Center:** $c_i = M_{ii}$
2. **Radius:** $R_i = \sum_{j \neq i} |M_{ij}|$
3. **Upper Bound:** $\rho_i = |c_i| + R_i$

### Step 3: Classify the Rows

Now, you split the indices of your system into two sets:

- **Stiff Set (S):** All indices $i$ where $\rho_i > \tau$.
- **Non-Stiff Set (N):** All indices $i$ where $\rho_i \leq \tau$.

### Step 4: Split the Jacobian and the ODE function into explicit and implicit parts

You can now create your additive split based on these sets. When you use the Gershgorin row-sum method, you are essentially categorizing each **equation** in your system as either "fast" or "slow." Simply put the "fast" rows into the stiff matrix and the "slow" rows into the non-stiff 
matrix.

- **M_stiff​**: Contains rows $i \in \mathcal{S}$ of the original Jacobian matrix $M$, and all other rows are zero.
- **M_non_stiff​**: Contains rows $i \in \mathcal{N}$ of the original Jacobian matrix $M$, and all other rows are zero.

- **f_impl(y)​**: Contains only indices of f(y) corresponding to the fast rows of the Jacobian, and all other rows are zeroed out.
- **f_expl(y)​**: Contains only indices of f(y) corresponding to the slow rows of the Jacobian, and all other rows are zeroed out.

The resulting batched f_impl and f_expl components can then be passed through to the normal kencarp5 method. The jacobian need not be rederived for f_impl, as we have already computed it as M_stiff.

### The Trade-off

The Gershgorin partition is a **conservative heuristic**. It might over-identify rows as stiff because it assumes the worst-case scenario (the disks are just bounds, not the actual eigenvalues). However, it is **incredibly fast** ($O(n^2)$ for dense, $O(nnz)$ for sparse) because it requires no matrix inversions or iterative solvers to perform the split.

This allows us to do Reduced-Order Solving, which is implemented as follows:

## 1. The "Identity" Save

When you perform an IMEX step, you aren't just solving $M_{\text{stiff}} \mathbf{y} = \mathbf{b}$. You are solving the system that comes from the implicit integration rule (like Backward Euler or a DIRK stage):


$$
(I - h\gamma M_{\text{stiff}}) \mathbf{y}_{n+1} = \text{rhs}
$$

If you have used Method A to zero out the "non-stiff" rows in $M_{\text{stiff}}$, look at what happens to those rows in the total matrix $A = (I - h\gamma M_{\text{stiff}})$:

- In a zeroed-out row $i$, the $M_{\text{stiff}}$ part is $0$.
- The $I$ (Identity) part remains $1$ on the diagonal.
- Therefore, the $i$-th row of the system is simply: $[0, \dots, 0, 1, 0, \dots, 0] \mathbf{y}_{n+1} = \text{rhs}_i$.

This means the variable $y_i$ is already "solved"—it is simply equal to the right-hand side. The system remains **square**, but it becomes **partially decoupled**.

## 2. Block Partitioning (The "Trimming" Logic)

To make this computationally efficient, you can reorder your variables so that all the non-stiff variables come first and all the stiff variables come last. Your system then takes on a **Block Lower Triangular** structure:


$$
\begin{bmatrix} I & 0 \\ \text{Coupling} & I - h\gamma M_{ss} \end{bmatrix} \begin{bmatrix} \mathbf{y}_{non-stiff} \\ \mathbf{y}_{stiff} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_{non-stiff} \\ \mathbf{b}_{stiff} \end{bmatrix}
$$

### The Workflow:

1. **Solve the Top Block:** $\mathbf{y}_{non-stiff} = \mathbf{b}_{non-stiff}$. (This is instant; no math required).
2. **Substitute into the Bottom Block:**


$$
(I - h\gamma M_{ss}) \mathbf{y}_{stiff} = \mathbf{b}_{stiff} - (\text{Coupling} \times \mathbf{y}_{non-stiff})
$$
3. **The Reduced Solve:** Now you only have to perform an **LU factorization** on the matrix $(I - h\gamma M_{ss})$.

This matrix is **square**, it's much smaller than the original $M$, and it represents only the "stiff-on-stiff" interactions.

## 3. When does this benefit disappear?

The only time this "trimming" strategy fails is if your **stiff variables are coupled to every other variable.** If every row has at least one "fast" interaction, you can't zero out any rows. But in physical systems (like a power grid or a large chemical network), stiffness is often localized to a small subset of "fast" components. In those cases, the size of the matrix you actually need to factorize might be only 10% or 20% of the total system size.

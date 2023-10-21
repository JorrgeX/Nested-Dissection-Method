import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse import lil_matrix
import scipy.sparse as sc
import scipy.sparse.linalg as la
import time

def create_poisson_matrix(n):
    # Create the diagonals for the Poisson matrix
    main_diag = -4 * np.ones(n**2)
    side_diag = np.ones(n**2 - 1)
    top_bottom_diag = np.ones(n**2 - n)
    
    # Create a sparse matrix with these diagonals
    diagonals = [main_diag, side_diag, side_diag, top_bottom_diag, top_bottom_diag]
    offsets = [0, 1, -1, n, -n]
    poisson_matrix = diags(diagonals, offsets, shape=(n**2, n**2), format='lil')
    
    # Convert to a dense matrix
    poisson_matrix = poisson_matrix.toarray()

    return poisson_matrix

n =  64 # 16, 32, 64
A = create_poisson_matrix(n)

n_half = n // 2

p1 = np.arange(n*n_half)
p2 = np.arange(n**2 - 1, n*(n_half + 1)-1, -1)
# p2 = np.arange(n*(n_half + 1), n**2)
p3 = np.arange(n*n_half, n*(n_half + 1))
perm = np.concatenate((p1, p2, p3))

A_reorder = A[perm[:, None], perm]
print("A_reorder:", A_reorder.shape)

left_range = p3[:len(p3) // 2]
right_range = p3[len(p3) // 2:]

A11 = A_reorder[:n*n_half, :n*n_half]
A13 = A_reorder[:n*n_half, n**2 - n:]
A31 = A_reorder[n**2 - n:, :n*n_half]
A22 = A_reorder[n*n_half:n**2 - n, n*n_half:n**2 - n]
A23 = A_reorder[n*n_half:n**2 - n, n**2 - n:]
A32 = A_reorder[n**2 - n:, n*n_half:n**2 - n]
A33 = A_reorder[n**2 - n:, n**2 - n:]

# checking the submatrix
# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
# ax1.spy(A11, marker = "s",markersize = 0.5)
# ax3.spy(A13, marker = "s",markersize = 0.5)
# ax5.spy(A22, marker = "s",markersize = 0.5)
# ax6.spy(A23, marker = "s",markersize = 0.5)
# ax7.spy(A31, marker = "s",markersize = 0.5)
# ax8.spy(A32, marker = "s",markersize = 0.5)
# ax9.spy(A33, marker = "s",markersize = 0.5)
# plt.show()

A11_inv = np.linalg.inv(A11)
S11 = np.dot(A31, np.dot(A11_inv, A13))
print("S11:", np.linalg.matrix_rank(S11))
Slr = S11[:n_half, n_half:]
Srl = S11[n_half:, :n_half]
print("Slr:", np.linalg.matrix_rank(Slr))
print("Srl:", np.linalg.matrix_rank(Srl))

A22_inv = np.linalg.inv(A22)
S22 = np.dot(A32, np.dot(A22_inv, A23))
print("S22:", np.linalg.matrix_rank(S22))
Slr = S22[:n_half, n_half:]
Srl = S22[n_half:, :n_half]


print("Slr:", np.linalg.matrix_rank(Slr, tol=1e-6))
print("Srl:", np.linalg.matrix_rank(Srl, tol=1e-12))

U, S, V = np.linalg.svd(Slr, full_matrices=False)
u, s, v = np.linalg.svd(Srl, full_matrices=False)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.arange(1, len(S) + 1), S, "o")
ax2.plot(np.arange(1, len(s) + 1), S, "o")
ax1.title.set_text("$S_{lr}$")
ax2.title.set_text("$S_{rl}$")
plt.show()

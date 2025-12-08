#Ramon Näf; 20-116-950

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse.linalg import lsmr, svds
from scipy.sparse import diags
import cv2

def get_normalization_matrix(x):
    """
    get_normalization_matrix Returns the transformation matrix used to normalize
    the inputs x
    Normalization corresponds to subtracting mean-position and positions
    have a mean distance of sqrt(2) to the center
    """
    # Input: x 3*N
    #
    # Output: T 3x3 transformation matrix of points

    # TO DO TASK:
    # --------------------------------------------------------------
    # Estimate transformation matrix used to normalize
    # the inputs x
    # --------------------------------------------------------------

    # Get centroid and mean-distance to centroid
    x = x.copy().astype(float)
    x /= x[2, :] 

    u = x[0, :]
    v = x[1, :]

    cx = np.mean(u)
    cy = np.mean(v)

    du = u - cx
    dv = v - cy

    d = np.sqrt(du**2 + dv**2)
    mean_d = np.mean(d)

    if mean_d < 1e-12:
        s = 1.0
    else:
        s = np.sqrt(2) / mean_d

    T = np.array([[s, 0, -s * cx],
                  [0, s, -s * cy],
                  [0, 0, 1]])

    return T


def eight_points_algorithm(x1, x2, normalize=True):
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm
    Inputs:
                    x1      3xN     homogeneous coordinates of matched points in view 1
                    x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
                    F       3x3     fundamental matrix
    """
    N = x1.shape[1]

    if normalize:
        # Construct transformation matrices to normalize the coordinates
        # TODO:
        T1 = get_normalization_matrix(x1)
        T2 = get_normalization_matrix(x2)

        # Normalize inputs
        # TODO:
        x1n = T1 @ x1
        x2n = T2 @ x2
    else:
        x1n, x2n = x1, x2

    # Construct matrix A encoding the constraints on x1 and x2
    # TODO:
    x1n = x1n / x1n[2, :]
    x2n = x2n / x2n[2, :]

    u1 = x1n[0, :]
    v1 = x1n[1, :]
    u2 = x2n[0, :]
    v2 = x2n[1, :]

    N = x1.shape[1]

    A = np.column_stack([u2 * u1, 
                        u2 * v1, 
                        u2, 
                        v2 * u1, 
                        v2 * v1, 
                        v2, 
                        u1,
                        v1,
                        np.ones(N)])

    # Solve for f using SVD
    # TODO:
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1, :]
    F = f.reshape(3, 3)

    # Enforce that rank(F)=2
    # TODO:
    U_f, S_f, Vt_f = np.linalg.svd(F)
    S_f[-1] = 0
    F_norm = U_f @ np.diag(S_f) @ Vt_f


    if normalize:
        # Transform F back 
        # TODO:
        F = T2.T @ F_norm @ T1
    else:
        F = F_norm

    return F


def right_epipole(F):
    """
    Computes the (right) epipole from a fundamental matrix F.
    (Use with F.T for left epipole.)
    """

    # The epipole is the null space of F (F * e = 0)
    # TODO
    _,_, Vt = np.linalg.svd(F)
    e = Vt[-1, :]

    if np.abs(e[2]) > 1e-12:
        e = e / e[2]

    return e


def plot_epipolar_line(im, F, x, e, ax = None):
    """
    Plot the epipole and epipolar line F*x=0 in an image. F is the fundamental matrix
    and x a point in the other image.
    """
    m, n = im.shape[:2]
    # TODO

    if ax is None:
        ax = plt

    a, b, c = F @ x

    ax.imshow(im, cmap="gray")

    if e is not None and np.abs(e[2]) > 1e-12:
        ex, ey = e[0] / e[2], e[1] / e[2]
        ax.plot(ex, ey, 'rx')

    if abs(b) > 1e-8:
        u0, u1 = 0, n - 1
        v0 = -(c + a * u0) / b
        v1 = -(c + a * u1) / b
    elif abs(a) >= 1e-12:
        u0 = u1 = -c / a
        v0, v1 = 0, m - 1

    ax.plot([u0, u1], [v0, v1])




def ransac(x1, x2, threshold, num_steps=1000, random_seed=42):
    if random_seed is not None:
        np.random.seed(random_seed)  # we are using a random seed to make the results reproducible

    # TODO setup variables
    assert x1.shape[1] == x2.shape[1]
    N = x1.shape[1]
    assert N >= 8, "Need at least 8 correspondences for 8-point algorithm"

    best_num_inliers = 0
    best_sample_indices = None

    for _ in range(num_steps):
        # TODO main loop
        idx = np.random.choice(N, 8, replace=False)

        try:
            F_candidate = eight_points_algorithm(x1[:, idx], x2[:, idx], normalize=True)
        except np.linalg.LinAlgError:
            continue

        d = np.abs(np.sum(x2 * (F_candidate @ x1), axis=0))

        inliers_mask = d < threshold
        num_inliers = np.sum(inliers_mask)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_sample_indices = idx

    if best_sample_indices is None:
        F = eight_points_algorithm(x1, x2, normalize=True)
        inliers = np.ones(N, dtype=bool)
        return F, inliers

    # TODO calculate initial inliers with with the best candidate points
    F_initial = eight_points_algorithm(x1[:, best_sample_indices],
                                       x2[:, best_sample_indices],
                                       normalize=True)
    d_initial = np.abs(np.sum(x2 * (F_initial @ x1), axis=0))
    initial_inliers = d_initial < threshold

    # TODO estimate F with all the inliers
    F_refined = eight_points_algorithm(x1[:, initial_inliers],
                                       x2[:, initial_inliers],
                                       normalize=True)
    
    # TODO find final inliers with F
    d_final = np.abs(np.sum(x2 * (F_refined @ x1), axis=0))
    final_inliers = d_final < threshold

    F, inliers = F_refined, final_inliers
    
    return F, inliers  # F is estimated fundamental matrix and inliers is an indicator (boolean) numpy array


def decompose_essential_matrix(E, x1, x2):
    """
    Decomposes E into a rotation and translation matrix using the
    normalized corresponding points x1 and x2.
    """

    # Fix left camera-matrix
    Rl = np.eye(3)
    tl = np.array([[0, 0, 0]]).T
    Pl = np.concatenate((Rl, tl), axis=1)
    
    # TODO: Compute possible rotations and translations
    U, S, Vt = np.linalg.svd(E)

    s = (S[0] + S[1]) / 2.0
    E_hat = U @ np.diag([s, s, 0.0]) @ Vt

    U, S, Vt = np.linalg.svd(E_hat)

    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    t = U[:, 2].reshape(3, 1)
    t1 = t
    t2 = -t

    # Four possibilities
    Pr = [np.concatenate((R1, t1), axis=1),
          np.concatenate((R1, t2), axis=1),
          np.concatenate((R2, t1), axis=1),
          np.concatenate((R2, t2), axis=1)]

    # Compute reconstructions for all possible right camera-matrices
    X3Ds = [infer_3d(x1[:, 0:1], x2[:, 0:1], Pl, x) for x in Pr]

    # Compute projections on image-planes and find when both cameras see point
    test = [np.prod(np.hstack((Pl @ np.vstack((X3Ds[i], [[1]])), Pr[i] @ np.vstack((X3Ds[i], [[1]])))) > 0, 1) for i in
            range(4)]
    test = np.array(test)
    idx = np.where(np.hstack((test[0, 2], test[1, 2], test[2, 2], test[3, 2])) > 0.)[0][0]

    # Choose correct matrix
    Pr = Pr[idx]

    return Pl, Pr


def infer_3d(x1, x2, Pl, Pr):
    # INFER3D Infers 3d-positions of the point-correspondences x1 and x2, using
    # the rotation matrices Rl, Rr and translation vectors tl, tr. Using a
    # least-squares approach.

    M = x1.shape[1]
    # Extract rotation and translation
    Rl = Pl[:3, :3]
    tl = Pl[:3, 3]
    Rr = Pr[:3, :3]
    tr = Pr[:3, 3]

    # Construct matrix A with constraints on 3d points
    row_idx = np.tile(np.arange(4 * M), (3, 1)).T.reshape(-1)
    col_idx = np.tile(np.arange(3 * M), (1, 4)).reshape(-1)

    A = np.zeros((4 * M, 3))
    A[:M, :3] = x1[0:1, :].T @ Rl[2:3, :] - np.tile(Rl[0:1, :], (M, 1))
    A[M:2 * M, :3] = x1[1:2, :].T @ Rl[2:3, :] - np.tile(Rl[1:2, :], (M, 1))
    A[2 * M:3 * M, :3] = x2[0:1, :].T @ Rr[2:3, :] - np.tile(Rr[0:1, :], (M, 1))
    A[3 * M:4 * M, :3] = x2[1:2, :].T @ Rr[2:3, :] - np.tile(Rr[1:2, :], (M, 1))

    A = sparse.csr_matrix((A.reshape(-1), (row_idx, col_idx)), shape=(4 * M, 3 * M))

    # Construct vector b
    b = np.zeros((4 * M, 1))
    b[:M] = np.tile(tl[0], (M, 1)) - x1[0:1, :].T * tl[2]
    b[M:2 * M] = np.tile(tl[1], (M, 1)) - x1[1:2, :].T * tl[2]
    b[2 * M:3 * M] = np.tile(tr[0], (M, 1)) - x2[0:1, :].T * tr[2]
    b[3 * M:4 * M] = np.tile(tr[1], (M, 1)) - x2[1:2, :].T * tr[2]

    # Solve for 3d-positions in a least-squares way
    w = lsmr(A, b)[0]
    x3d = w.reshape(M, 3).T

    return x3d

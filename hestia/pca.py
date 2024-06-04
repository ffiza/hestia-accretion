import numpy as np


def PCA_matrix(GasPos, GasVel, r_inner, M31_pos=False):
    """
    If M31_pos is passed, prioritize it being on the southern hemisphere from
    MW over aligning with galactic angular momentum.
    """
    inner_gas = np.where(
        GasPos[:, 0]**2 + GasPos[:, 1]**2 + GasPos[:, 2]**2 < r_inner**2)
    GasPos = GasPos[inner_gas]
    GasVel = GasVel[inner_gas]
    # We calculate covariance matrix and diagonalize it. The eigenvectors are
    # the galaxy's principal axes
    covMatrix = np.cov(np.transpose(GasPos))
    eigenval, eigenvect = np.linalg.eig(covMatrix)
    # Eigenvalues are not ordered; we make it so rot_matrix has eigenvectors
    # as columns ordered from highest eigenvalue to lowest:
    eig1 = eigenval.argmax()
    eig3 = eigenval.argmin()
    eig2 = 3 - eig1 - eig3
    rot_matrix = np.array([eigenvect[:, eig1],
                           eigenvect[:, eig2],
                           eigenvect[:, eig3]])
    rot_matrix = np.transpose(rot_matrix)

    GasPos = np.dot(GasPos, rot_matrix)

    GasPos_x = GasPos[:, 0]
    GasPos_y = GasPos[:, 1]
    GasVel_x = GasVel[:, 0]
    GasVel_y = GasVel[:, 1]

    jz = GasPos_x * GasVel_y - GasPos_y * GasVel_x

    if M31_pos:
        M31_rot = np.dot(M31_pos, rot_matrix)
        if M31_rot[-1] > 0:
            rot_matrix[:, 0] = - rot_matrix[:, 0]
            rot_matrix[:, 2] = - rot_matrix[:, 2]
    elif np.sum(jz) > 0:
        # The North Galactic Pole is in the opposite direction of
        # the angular momentum, so we leave j in -z.
        # We invert first and last row (x and z) from
        # the rot_matrix which is equivalent to rotating around the y axis.
        rot_matrix[:, 0] = - rot_matrix[:, 0]
        rot_matrix[:, 2] = - rot_matrix[:, 2]

    return rot_matrix

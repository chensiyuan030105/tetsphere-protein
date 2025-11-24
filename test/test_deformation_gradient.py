import numpy as np
import argparse
import torch

def compute_G_matrix(verts_init, faces):
    # Compute gradient operator matrix
    T = faces.shape[0]
    
    # Print dimensions at key steps
    print(f"Dimensions of faces: {faces.shape}")  # Print dimensions of faces

    if type(verts_init) == np.ndarray:
        verts_init = torch.from_numpy(verts_init).to(torch.float64)
    print(f"Dimensions of verts_init (converted to tensor): {verts_init.shape}")  # Print dimensions of verts_init

    Gd = torch.zeros([4, 3], device=verts_init.device).to(torch.float64)
    Gd[0, :] = -1.0
    Gd[1, 0] = 1.0
    Gd[2, 1] = 1.0
    Gd[3, 2] = 1.0  # 4 x 3

    print(f"Dimensions of Gd (gradient matrix): {Gd.shape}")  # Print dimensions of Gd
    print(f"Gd values:\n{Gd}")  # Print Gd values

    X = verts_init[faces]  # T x 4 x 3
    print(f"Dimensions of X (vertices per face): {X.shape}")  # Print dimensions of X
    print(f"X values:\n{X[:5]}")  # Print first 5 values of X for inspection

    X = X.transpose(1, 2)  # T x 3 x 4
    print(f"Dimensions of X after transpose: {X.shape}")  # Print dimensions of X after transpose
    print(f"X after transpose values:\n{X[:5]}")  # Print first 5 values of X after transpose

    dX = torch.matmul(X, Gd.unsqueeze(0).expand(T, -1, -1))  # T x 3 x 3
    print(f"Dimensions of dX (change in coordinates): {dX.shape}")  # Print dimensions of dX
    print(f"dX values:\n{dX[:5]}")  # Print first 5 values of dX

    dX_inv = torch.inverse(dX)  # T x 3 x 3
    print(f"Dimensions of dX_inv (inverse of dX): {dX_inv.shape}")  # Print dimensions of dX_inv
    print(f"dX_inv values:\n{dX_inv[:5]}")  # Print first 5 values of dX_inv

    G = torch.zeros([T, 9, 12], device=verts_init.device).to(torch.float64)
    print(f"Initial dimensions of G: {G.shape}")  # Print dimensions of G before loop
    print(f"Initial values of G:\n{G[:5]}")  # Print initial G values

    for dofi in range(12):
        E = torch.zeros([3, 4], device=verts_init.device).to(torch.float64)
        E[dofi % 3, dofi // 3] = 1.0

        Z = E @ Gd
        R = Z.unsqueeze(0).expand(T, -1, -1) @ dX_inv
        G[:, :, dofi] = R.view(T, -1)

        # Print the dimensions and values of R and how it is assigned to G
        print(f"Dimensions of R (deformation matrix for dofi={dofi}): {R.shape}")  # Print dimensions of R
        print(f"R values for dofi={dofi}:\n{R[:5]}")  # Print first 5 values of R for this dofi

    print(f"Final dimensions of G: {G.shape}")  # Print dimensions of G after the loop
    print(f"Final values of G:\n{G[:5]}")  # Print first 5 values of the final G matrix
    return G.numpy()  # T x 9 x 12

# Main function to parse command line arguments and call the test function
if __name__ == "__main__":
    # Create argument parser to handle input file paths from command line
    parser = argparse.ArgumentParser(description="Test vertex and element files")
    parser.add_argument("vtx_file", type=str, help="Path to the vertex .npy file")
    parser.add_argument("elem_file", type=str, help="Path to the element .npy file")
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Load vertex data from the .npy file
    verts = np.load(args.vtx_file)
    
    # Load element data from the .npy file
    elems = np.load(args.elem_file)

    # Call the function with the parsed file paths
    compute_G_matrix(verts, elems)

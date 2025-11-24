import numpy as np
import argparse

# Function to load vertex and element files and print related information
def test_verts_faces(vtx_file, elem_file):
    # Load vertex data from the .npy file
    verts = np.load(vtx_file)
    
    # Load element data from the .npy file
    elems = np.load(elem_file)

    # Output vertex information
    num_verts = verts.shape[0]  # Number of vertices
    vert_dim = verts.shape[1] if len(verts.shape) > 1 else 1  # Dimension of each vertex (e.g., 3 for 3D vertices)

    # Output element information
    num_elems = elems.shape[0]  # Number of elements
    elem_size = elems.shape[1] if len(elems.shape) > 1 else 1  # Number of vertices per element (typically 4 for tetrahedra)

    # Print out the results
    print(f"Number of vertices: {num_verts}")
    print(f"Dimension of each vertex: {vert_dim}")
    print(f"Number of elements: {num_elems}")
    print(f"Number of vertices per element: {elem_size}")

    # Print the first few vertex coordinates as a sample
    print("\nFirst 5 vertices (sample):")
    print(verts[:5])  # Show the first 5 vertices

    # Print the first few elements as a sample
    print("\nFirst 5 elements (sample):")
    print(elems[:5])  # Show the first 5 elements

# Main function to parse command line arguments and call the test function
if __name__ == "__main__":
    # Create argument parser to handle input file paths from command line
    parser = argparse.ArgumentParser(description="Test vertex and element files")
    parser.add_argument("vtx_file", type=str, help="Path to the vertex .npy file")
    parser.add_argument("elem_file", type=str, help="Path to the element .npy file")
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Call the test function with the parsed file paths
    test_verts_faces(args.vtx_file, args.elem_file)

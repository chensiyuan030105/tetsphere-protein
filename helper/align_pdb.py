from Bio import PDB
import sys

def align_pdbs(pdb_file_1, pdb_file_2, output_file):
    # Parse the PDB files
    parser = PDB.PDBParser(QUIET=True)
    structure_1 = parser.get_structure('protein_1', pdb_file_1)
    structure_2 = parser.get_structure('protein_2', pdb_file_2)
    
    # Get the first model from both structures (PDB can contain multiple models, we select the first one)
    model_1 = structure_1[0]
    model_2 = structure_2[0]
    
    # Select atoms that are common to both models
    atoms_1 = list(model_1.get_atoms())
    atoms_2 = list(model_2.get_atoms())

    # Make sure both atom lists have the same length
    # Filter atoms based on the same residue type and atom name
    atoms_1_filtered = []
    atoms_2_filtered = []
    
    for atom_1 in atoms_1:
        for atom_2 in atoms_2:
            if atom_1.get_name() == atom_2.get_name() and atom_1.get_parent().get_resname() == atom_2.get_parent().get_resname():
                atoms_1_filtered.append(atom_1)
                atoms_2_filtered.append(atom_2)
                break

    if len(atoms_1_filtered) != len(atoms_2_filtered):
        raise ValueError("The number of matching atoms is different between the two structures.")

    print("atoms_1_filtered =", atoms_1_filtered)
    print(" ")
    print("atoms_2_filtered =", atoms_2_filtered)
    # Use the Superimposer to align the filtered atoms
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(atoms_1_filtered, atoms_2_filtered)
    super_imposer.apply(atoms_2_filtered)  # Apply the alignment to model_2
    
    # Save the aligned structure to the output PDB file
    io = PDB.PDBIO()
    io.set_structure(structure_2)
    io.save(output_file)
    
    print(f"Alignment complete. Updated structure saved as {output_file}")

if __name__ == "__main__":
    # Check if the script is executed with the correct number of arguments
    if len(sys.argv) != 4:
        print("Usage: python align_pdbs.py <pdb_file_1> <pdb_file_2> <output_file>")
        sys.exit(1)
    
    # Get the input and output file names from command line arguments
    pdb_file_1 = sys.argv[1]
    pdb_file_2 = sys.argv[2]
    output_file = sys.argv[3]
    
    # Call the function to align the PDB files
    align_pdbs(pdb_file_1, pdb_file_2, output_file)


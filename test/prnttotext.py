def write_coordinates_to_file(filename, *coordinates):
    """Writes sets of coordinates to a text file, each line representing one shape."""
    with open(filename, "a") as file:
        file.write(",".join(map(str, coordinates)) + "\n")

# Usage Example:
write_coordinates_to_file("shapes.txt", 1, 2, 3, 4)       # Shape 1
write_coordinates_to_file("shapes.txt", 5, 6, 7, 8, 9, 10) # Shape 2
write_coordinates_to_file("shapes.txt", 11, 12, 13)        # Shape 3
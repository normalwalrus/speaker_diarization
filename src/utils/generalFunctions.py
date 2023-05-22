import os

def get_file_list_from_dir(datadir, filetype = '.wav'):
    """
    Get list of files from a given directory, able to specifies the filetype to look for

    Parameters
    ----------
    datadir : String
        Path of the directory to get the file list from
    filetype : String
        The file extension the function is looking out for
        
    Returns
    ----------
    data_files : List
        List of strings with the names of all the file names with the given file extensions
        
    """
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith(filetype), all_files))
    return data_files

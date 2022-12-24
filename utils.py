import pickle
import os


def saveVar(var, filePath):
    """
    saveVar saves a var to a specified filePath
    INPUT:
    var: the variable to be saved
    filePath: the filepath you want to save the var, for example data/.../var.pckl. If the path does not exist, it creates it
    """

    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):  # makes the directory if it does not exist
        os.makedirs(directory)

    # uses pickle to serialize and save the variable var
    pickleOut = open(filePath, "wb")
    pickle.dump(var, pickleOut)
    pickleOut.close()


def loadVar(filePath):
    """
    LOADVAR loads a var from a specified filePath
    INPUT:
    filePath: where the variable is
    OUTPUT:
    var: the variable loaded
    """

    pickleIn = open(filePath, "rb")
    var = pickle.load(pickleIn)
    pickleIn.close()

    return var


def notExist(arg1, arg2):
    """
    checks if arguments exist in workspace
    args
        arg1: str
        arg2: str
    returns
        True or False
    """
    if (arg1 not in locals()) | (arg2 not in locals()):
        return True




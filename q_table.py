import numpy as np

# Stores only one value that it returns on any key
class FinalState():
    def __init__(self):
        self.value = 0
    def __getitem__(self, key):
        return self.value
    def __setitem__(self, key, value):
        self.value = value
    def values(self):
        return [self.value]
    def __repr__(self):
        return "{any: " + str(self.value) + "}"

# Q_table[state][move]
class QTable(dict):
    def __getitem__(self, key):
        self.__check_key(key)
        return super().__getitem__(self.__to_tuple(key))
    def __setitem__(self, key, value):
        self.__check_key(key)
        super().__setitem__(self.__to_tuple(key), value)

    def __missing__(self, key):
        arr = self.__to_array(key)
        non_zero_locations = list(zip(*np.nonzero(arr == 0)))

        # check if game has ended
        if (len(non_zero_locations) == 0 or # check if all the positions are filled
            np.any(np.sum(arr, axis=0) == 3) or # check rows and cols
            np.any(np.sum(arr, axis=1) == 3) or
            np.any(np.sum(arr, axis=0) == -3) or 
            np.any(np.sum(arr, axis=1) == -3) or
            arr[[0,1,2],[0,1,2]].sum() == 3 or # check diagonals
            arr[[0,1,2],[2,1,0]].sum() == 3 or 
            arr[[0,1,2],[0,1,2]].sum() == -3 or 
            arr[[0,1,2],[2,1,0]].sum() == -3):
            ret = self[arr] = FinalState()
            return ret
        ret = self[arr] = {loc: 0 for loc in non_zero_locations}
        return ret

    def __check_key(self, key):
        if not isinstance(key, np.ndarray) and not isinstance(key, tuple):
            raise(TypeError("QTable only accepts np.ndarray of as keys"))
        if isinstance(key, tuple) and (len(key) != 9):
            raise(ValueError(f"QTable only tuples of length 9, your length {len(key)}"))
        if isinstance(key, np.ndarray) and (key.size != 9):
            raise(ValueError(f"QTable only accepts np.ndarray of size 9, your size {key.size}"))
    def __to_array(self, key):
        if isinstance(key, tuple):
            key = np.array(key).reshape(3, 3)
        return key
    def __to_tuple(self, key):
        if isinstance(key, np.ndarray):
            key = tuple(key.reshape(-1))
        return key
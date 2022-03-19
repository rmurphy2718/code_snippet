class IdMap:
    def __init__(self) -> None:
        self._id_to_name = dict()
        self._name_to_id = dict()
        self._curr = -1

    def get_name(self, id):
        return self._id_to_name[id]

    def get_id(self, name):
        if name not in self._name_to_id:
            self._curr += 1
            self._name_to_id[name] = self._curr
        return self._name_to_id[name]
